#ifndef CAMERA_MODULE_H
#define CAMERA_MODULE_H

#include <QObject>
#include <QTimer>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

/**
 * @class CameraModule
 * @brief 面向对象封装的相机管理模块。
 *
 * 设计思路：
 *  - Qt 主线程中创建 CameraModule 对象，其他类只需要通过公开接口即可完成相机的打开、关闭和逐帧采集；
 *  - 该类内部完全掌握 cv::VideoCapture、计时器状态等资源，避免主窗口直接操控底层句柄造成的耦合；
 *  - 通过 Qt 信号槽向外发送采集到的新帧，从而允许图像处理模块或者 UI 控件订阅画面；
 *  - 尽可能在注释中明确每个步骤的细节，方便后续维护人员理解相机生命周期管理的全过程。
 */
class CameraModule : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数。
     * @param parent Qt 对象树中的父对象，确保生命周期自动管理。
     */
    explicit CameraModule(QObject *parent = nullptr);

    /**
     * @brief 配置用于周期触发采集的 QTimer。
     *
     * 在主线程中往往已经存在一个 QTimer（例如 30ms 触发一次），
     * 因此通过该接口可以复用外部计时器，而不是在类内部随意 new。
     * 传入 nullptr 表示暂不使用计时器，仅需手动调用 captureFrame()。
     */
    void setCaptureTimer(QTimer *timer);

    /**
     * @brief 控制是否向外广播每帧图像。
     *
     * 默认关闭，以避免在未订阅 frameCaptured 信号时仍然为每帧执行一次深拷贝。
     */
    void setFrameSignalEnabled(bool enabled);

    /**
     * @brief 打开指定索引的摄像头。
     * @param deviceIndex 在 Qt 端通常由 QMediaDevices::videoInputs() 提供。
     * @param width 希望设置的帧宽；传入 0 表示沿用设备默认值。
     * @param height 希望设置的帧高；传入 0 表示沿用设备默认值。
     * @return true 表示打开成功，false 表示失败。
     */
    bool openCamera(int deviceIndex, int width = 0, int height = 0);

    /**
     * @brief 关闭当前摄像头并释放底层资源。
     * @note 调用后 isCameraOpen() 会返回 false。
     */
    void closeCamera();

    /**
     * @brief 判断摄像头是否处于打开状态。
     */
    bool isCameraOpen() const;

    /**
     * @brief 抓取一帧图像。
     *
     * 当外部不使用计时器驱动时，可以直接调用该函数主动抓帧；
     * 若使用 setCaptureTimer() 注册了计时器且连接了 onTimeout() 槽函数，
     * 那么内部也会在定时器触发时主动调用本函数。
     *
     * @param outFrame 输出参数，用于接收采集到的 BGR 彩色帧。
     * @return true 表示成功抓取到有效帧，false 表示抓取失败或者摄像头未打开。
     */
    bool captureFrame(cv::Mat &outFrame);

    /**
     * @brief 设置底层 VideoCapture 的属性（例如分辨率、帧率）。
     * @param propertyId OpenCV CAP_PROP_* 常量。
     * @param value      目标属性值。
     * @return 设置成功返回 true，失败返回 false。
     */
    bool setProperty(int propertyId, double value);

signals:
    /**
     * @brief 摄像头成功开启时发出的信号。
     * @details 方便主窗口更新按钮文字、状态灯等 UI。
     */
    void cameraOpened();

    /**
     * @brief 摄像头关闭时发出的信号。
     */
    void cameraClosed();

    /**
     * @brief 成功抓取到一帧图像时发出的信号。
     *
     * @param frame 以深拷贝的方式传递给接收者，避免共享生命周期带来的线程安全问题。
     */
    void frameCaptured(const cv::Mat &frame);

private slots:
    /**
     * @brief 外部计时器触发时自动抓取一帧。
     *
     * 该槽函数不会直接暴露给 UI，而是作为 CameraModule 的内部驱动逻辑：
     * 每当定时器 timeout()，都会调用 captureFrame() 并广播 frameCaptured()。
     */
    void onTimeout();

private:
    cv::VideoCapture m_capture; ///< OpenCV 提供的底层采集对象。
    bool m_isOpen = false;      ///< 当前摄像头状态缓存，避免频繁访问 VideoCapture::isOpened()。
    QTimer *m_timer = nullptr;  ///< 外部传入的定时器指针，不负责释放。
    bool m_emitFrameSignal = false; ///< 是否发送 frameCaptured 信号，避免无订阅情况下的额外拷贝。
};

#endif // CAMERA_MODULE_H
