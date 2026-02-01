#include "camera_module.h"

#include <QDebug>

CameraModule::CameraModule(QObject *parent)
    : QObject(parent)
{
    // 在构造函数中仅执行轻量操作，例如初始化标志位。
    // 真正的硬件打开动作交由 openCamera() 完成，以便按需调用。
}

void CameraModule::setCaptureTimer(QTimer *timer)
{
    if (m_timer == timer) {
        return;
    }

    if (m_timer) {
        // 如果之前已经绑定了计时器，需要先断开旧连接，避免重复触发。
        disconnect(m_timer, &QTimer::timeout, this, &CameraModule::onTimeout);
    }

    m_timer = timer;
    if (m_timer) {
        // 计时器成功传入后立即建立连接，让 onTimeout() 成为周期驱动的入口。
        connect(m_timer, &QTimer::timeout, this, &CameraModule::onTimeout);
    }
}

void CameraModule::setFrameSignalEnabled(bool enabled)
{
    m_emitFrameSignal = enabled;
}

bool CameraModule::openCamera(int deviceIndex, int width, int height)
{
    if (m_isOpen) {
        // 若已经处于开启状态，先安全关闭，避免重复占用设备句柄。
        closeCamera();
    }

    // VideoCapture::open 会尝试获取操作系统中的摄像头资源。
    if (!m_capture.open(deviceIndex)) {
        qWarning() << "CameraModule::openCamera -> 打开相机失败" << deviceIndex;
        m_isOpen = false;
        return false;
    }

    // 如果用户传入了正数尺寸，则尝试修改采集分辨率。
    if (width > 0) {
        m_capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    }
    if (height > 0) {
        m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }

    m_isOpen = m_capture.isOpened();
    if (m_isOpen) {
        emit cameraOpened();
    } else {
        qWarning() << "CameraModule::openCamera -> 虽然调用 open 成功但 isOpened 返回 false";
    }
    return m_isOpen;
}

void CameraModule::closeCamera()
{
    if (!m_isOpen) {
        return;
    }

    m_capture.release();
    m_isOpen = false;
    emit cameraClosed();
}

bool CameraModule::isCameraOpen() const
{
    return m_isOpen;
}

bool CameraModule::captureFrame(cv::Mat &outFrame)
{
    if (!m_isOpen) {
        qWarning() << "CameraModule::captureFrame -> 相机未打开";
        return false;
    }

    if (!m_capture.read(outFrame) || outFrame.empty()) {
        qWarning() << "CameraModule::captureFrame -> 读取到空帧";
        return false;
    }

    if (m_emitFrameSignal && receivers(SIGNAL(frameCaptured(cv::Mat)))) {
        emit frameCaptured(outFrame.clone());
    }
    return true;
}

bool CameraModule::setProperty(int propertyId, double value)
{
    if (!m_isOpen) {
        qWarning() << "CameraModule::setProperty -> 相机未打开";
        return false;
    }

    const bool ok = m_capture.set(propertyId, value);
    if (!ok) {
        qWarning() << "CameraModule::setProperty -> 设置失败" << propertyId << value;
    }
    return ok;
}

void CameraModule::onTimeout()
{
    cv::Mat frame;
    captureFrame(frame);
    // captureFrame 会负责在成功采集后发送 frameCaptured 信号，因此这里无需额外处理。
}
