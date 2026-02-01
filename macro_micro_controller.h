#ifndef MACRO_MICRO_CONTROLLER_H
#define MACRO_MICRO_CONTROLLER_H

#include <QObject>
#include <QString>
#include <QSize>
#include <vector>
#include <opencv2/core.hpp>

class MicroArm;
class MacroArm;
class CameraModule;
class ImageProcessor;
class QSerialPort;

/**
 * @class MacroMicroController
 * @brief 宏观机械臂与微动机械臂的协调控制中心。
 *
 * 在显微操作场景中，通常需要将宏观臂用于粗定位、微动臂用于精调整，
 * 并结合相机反馈完成闭环控制。MacroMicroController 通过组合上述模块
 *（MicroArm、MacroArm、CameraModule、ImageProcessor），在统一入口下对
 * 各类跨模块操作进行封装，例如视觉坐标到机械臂坐标的映射、RCM 约束、
 * 以及宏微协同调度策略等。
 */
class MacroMicroController : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造协调控制器，默认不绑定任何硬件或视觉组件。
     * @param parent Qt 对象树父节点。
     */
    explicit MacroMicroController(QObject *parent = nullptr);

    /**
     * @brief 注册微动机械臂实例，供后续闭环补偿调用。
     * @param microArm 微动机械臂通信封装指针。
     */
    void attachMicroArm(MicroArm *microArm);

    /**
     * @brief 注册宏观机械臂实例，用于执行大范围移动。
     * @param macroArm 宏观机械臂通信封装指针。
     */
    void attachMacroArm(MacroArm *macroArm);

    /**
     * @brief 绑定相机模块，便于读取当前视场中的几何信息。
     * @param camera 相机控制封装指针。
     */
    void attachCamera(CameraModule *camera);

    /**
     * @brief 绑定图像处理模块，获取二值化、边缘等视觉结果。
     * @param processor 图像处理线程封装指针。
     */
    void attachImageProcessor(ImageProcessor *processor);

    /**
     * @brief 绑定俯仰姿态串口，供 RCM 调整时发送命令。
     * @param serial 俯仰控制所用的 QSerialPort。
     */
    void attachPitchSerialPort(QSerialPort *serial);

    /**
     * @brief 使用视觉坐标更新机械臂的映射矩阵。
     * @param pixel 视觉像素坐标 (u,v)。
     * @param macroPose 宏观臂末端的笛卡尔位置 (x,y,z,rx,ry,rz)。
     */
    void updateVisionToRobotMapping(const cv::Point2d &pixel, const cv::Vec6d &macroPose);

    /**
     * @brief 根据像素误差自动调整微动臂位置，实现闭环对准。
     * @param pixelError 当前像素偏差 (du,dv)。
     */
    void compensateUsingMicroArm(const cv::Point2d &pixelError);

    /**
     * @brief 执行 RCM（Remote Center of Motion）约束下的组合运动。
     *
     * @param pivotInCamera 旋转中心在相机坐标系下的位置。
     * @param deltaAlpha    期望的旋转角度（弧度）。
     */
    void performRCMMotion(const cv::Point3d &pivotInCamera, double deltaAlpha);

    cv::Mat fitYawCircleMapping(const std::vector<cv::Vec4d> &pts,
                                double *cx = nullptr,
                                double *cy = nullptr,
                                double *r = nullptr,
                                double *phi = nullptr,
                                int *sgnOut = nullptr,
                                double *rmseAlpha = nullptr,
                                bool print = true) const;

    bool executeRCMPitchAlignment(double alphaStartDeg,
                                  double alphaTargetDeg,
                                  double linkLength,
                                  const cv::Vec6i &macroPose,
                                  QString *errorMessage = nullptr,
                                  int minZSafety = 100000);

    bool executeYawArcMove(int currentAngleMilli,
                           int targetAngleMilli,
                           const cv::Mat &alphaMapping,
                           int wristRx,
                           int wristRy,
                           const cv::Mat &homographyCalib,
                           double alphaCalibrationRad,
                           double imageCenterU,
                           double imageCenterV,
                           cv::Mat &updatedHomography,
                           cv::Matx22d &updatedLinear,
                           QString *errorMessage = nullptr);

    cv::Mat rotateTargetAboutUvCenter(const cv::Mat &H_in,
                                      double beta,
                                      double u0,
                                      double v0) const;

    /**
     * @brief 对外触发安全停止，请求所有运动逻辑尽快退出。
     */
    void requestStop();

signals:
    /**
     * @brief 成功执行一次宏微协同动作后发出的信号。
     */
    void motionExecuted();

    void safetyWarningRaised(const QString &message);

private:
    MicroArm *m_microArm = nullptr;          ///< 微动机械臂控制指针。
    MacroArm *m_macroArm = nullptr;          ///< 宏观机械臂控制指针。
    CameraModule *m_camera = nullptr;        ///< 相机模块指针（目前预留）。
    ImageProcessor *m_processor = nullptr;   ///< 图像处理模块指针（目前预留）。
    QSerialPort *m_pitchSerial = nullptr;    ///< 俯仰姿态串口通信对象。

    cv::Matx22d m_pixelToMicroArm{1, 0, 0, 1}; ///< 像素 -> 微动臂平面坐标映射矩阵。
    bool m_stopRequested = false;              ///< 外部请求停止的标志位。

    /// 构造 movep_canfd 指令并发送给宏观臂。
    bool sendMovePCanfd(int x, int y, int z, int rx, int ry, int rz,
                        bool follow, int trajectoryMode, int radio);
    /// 构造 movec 指令并发送给宏观臂。
    bool sendMoveC(int via_x, int via_y, int via_z, int via_rx, int via_ry, int via_rz,
                   int x, int y, int z, int rx, int ry, int rz,
                   int v, int r = 0, int loop = 0, bool trajectoryConnect = false);
    /// 轮询事件循环的非阻塞延时辅助函数。
    static void spinDelay(int milliseconds);
};

#endif // MACRO_MICRO_CONTROLLER_H
