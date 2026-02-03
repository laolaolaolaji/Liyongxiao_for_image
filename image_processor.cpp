#include "image_processor.h"

#include <QMutexLocker>

ImageProcessor::ImageProcessor(QObject *parent)
    : QThread(parent)
{
}

ImageProcessor::~ImageProcessor()
{
    requestStop();
    wait();
}

void ImageProcessor::setProcessor(const Processor &processor)
{
    QMutexLocker locker(&m_mutex);
    m_processor = processor;
}

void ImageProcessor::enqueueFrame(const FrameRequest &request)
{
    QMutexLocker locker(&m_mutex);

    // 控制队列长度，避免画面延迟堆积。
    if (static_cast<int>(m_queue.size()) >= m_maxQueueSize) {
        m_queue.pop_front();
    }

    // 直接使用浅拷贝，将昂贵的深拷贝留给真正需要修改数据的处理环节。
    // Mat/Vec 等类型内部使用引用计数，跨线程传递时持有自己的引用，不会被提前释放。
    FrameRequest copy = request;
    m_queue.emplace_back(std::move(copy));
    m_waitCondition.wakeOne();
}

void ImageProcessor::requestStop()
{
    QMutexLocker locker(&m_mutex);
    m_stopRequested = true;
    m_waitCondition.wakeOne();
}

void ImageProcessor::run()
{
    for (;;) {
        FrameRequest request;
        Processor processorCopy;

        {
            QMutexLocker locker(&m_mutex);
            if (m_stopRequested && m_queue.empty()) {
                break;
            }

            if (m_queue.empty()) {
                m_waitCondition.wait(&m_mutex);
                if (m_queue.empty()) {
                    continue;
                }
            }

            request = m_queue.front();
            m_queue.pop_front();
            if (!m_queue.empty()) {
                // 只处理最新帧，避免高分辨率下队列堆积导致卡顿。
                request = m_queue.back();
                m_queue.clear();
            }
            processorCopy = m_processor;
        }

        if (!processorCopy) {
            // 没有设置处理回调时直接将原图回传，避免帧被悄悄丢弃。
            ProcessedImage passthrough;
            passthrough.frame = request.frame;
            passthrough.displayMode = request.displayMode;
            emit processedFrame(passthrough);
            continue;
        }

        ProcessedImage result = processorCopy(request);
        emit processedFrame(result);
    }
}
