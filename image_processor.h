#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <QMetaType>
#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#include <functional>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

Q_DECLARE_METATYPE(cv::Mat)

/**
 * @class ImageProcessor
 * @brief 专门用于图像处理的后台线程。
 *
 * 由于图像处理往往计算量较大，直接在 UI 线程执行会导致界面卡顿。
 * ImageProcessor 通过继承 QThread 在单独线程中循环取出待处理帧，
 * 并在处理完成后通过 processedFrame() 信号异步通知主线程。
 */
class ImageProcessor : public QThread
{
    Q_OBJECT
public:
    /**
     * @brief 指定处理线程向界面回传的图像类型。
     */
    enum class DisplayMode
    {
        Original,    ///< 显示原始画面（经过必要的预处理）。
        Grayscale,   ///< 显示灰度图。
        ClaheGray,   ///< 显示 CLAHE 增强后的灰度图。
        Binary,      ///< 显示二值化结果并覆盖特征点。
        Edges        ///< 显示轮廓边缘并覆盖特征点。
    };

    /**
     * @brief 前端线程传递给图像处理线程的请求对象。
     */
    struct FrameRequest
    {
        cv::Mat frame;              ///< 待处理的原始图像帧。
        bool applyCalibration = false; ///< 是否执行去畸变矫正。
        cv::Mat cameraMatrix;       ///< 去畸变所需的相机内参矩阵。
        cv::Mat distCoeffs;         ///< 相机畸变系数。
        bool flipImage = false;     ///< 是否需要执行上下左右翻转。
        cv::Mat background;         ///< 背景参考图，用于前景提取。
        DisplayMode displayMode = DisplayMode::Original; ///< 当前期望展示的图像类型。
    };

    /**
     * @brief 图像处理线程返回给主线程的结果。
     */
    struct ProcessedImage
    {
        cv::Mat frame;              ///< 处理后的彩色图像，用于界面显示。
        cv::Point tipPosition{-1, -1}; ///< 检测到的针尖像素坐标。
        bool hasTip = false;        ///< 是否成功检测到针尖。
        cv::Point2f filteredTip{-1.0f, -1.0f}; ///< 平滑后的针尖亚像素坐标，便于差分记录。
        cv::Point2f measuredTip{-1.0f, -1.0f}; ///< 当前帧直接检测到的针尖亚像素坐标。
        bool hasMeasuredTip = false;           ///< 标记当前帧是否存在可用的原始针尖检测结果。
        DisplayMode displayMode = DisplayMode::Original; ///< 回传时使用的图像类型，方便主线程感知。
    };

    using Processor = std::function<ProcessedImage(const FrameRequest &)>; ///< 自定义处理函数原型。

    explicit ImageProcessor(QObject *parent = nullptr);
    ~ImageProcessor() override;

    /**
     * @brief 设置实际执行图像处理逻辑的回调函数。
     *
     * 允许上层在运行时更换处理算法（例如边缘检测、标定等），
     * 从而实现模块之间的低耦合。
     */
    void setProcessor(const Processor &processor);

    /**
     * @brief 将一帧图像加入等待队列。
     *
     * @note 该函数会在内部执行深拷贝，确保原始帧可以在主线程安全释放。
     */
    void enqueueFrame(const FrameRequest &request);

    /**
     * @brief 请求线程安全退出。
     */
    void requestStop();

signals:
    /**
     * @brief 处理完成后的图像结果。
     */
    void processedFrame(const ProcessedImage &result);

protected:
    /**
     * @brief QThread 的线程入口。
     */
    void run() override;

private:
    Processor m_processor;               ///< 当前使用的图像处理函数。
    std::deque<FrameRequest> m_queue;    ///< 待处理帧队列。
    QMutex m_mutex;                      ///< 队列互斥锁。
    QWaitCondition m_waitCondition;      ///< 条件变量，用于阻塞线程。
    bool m_stopRequested = false;        ///< 停止标志。
    int m_maxQueueSize = 2;              ///< 队列长度上限，防止积压导致延迟。
};

Q_DECLARE_METATYPE(ImageProcessor::ProcessedImage)

#endif // IMAGE_PROCESSOR_H
