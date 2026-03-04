#ifndef MICROMANIPULATOR_H
#define MICROMANIPULATOR_H

#include <QMainWindow>
// #include "ImagePocessingThread.h"

#include <QTimer>
#include <QButtonGroup>
#include <QTime>
#include <QKeyEvent>
#include <QSet>
#include <QCameraDevice>
#include <QMediaDevices>
#include <QList>
#include <Qcamera>
#include <QThread>
#include <QCloseEvent>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QDebug>
#include <QByteArray>
#include <QSerialPort>
#include <QPoint>
#include <QFile>
#include <QTextStream>
#include <QString>
#include <windows.h> // 包含WinAPI头文件
#include <QMessageBox>
#include <QTcpSocket>    // 包含TCP通信相关头文件
#include <QNetworkInterface>
// #include "rm_define.h"
// #include "rm_interface.h"
#include <filesystem>
#include "jsonexplain.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include "camera_module.h"
#include "image_processor.h"
#include "macro_arm.h"
#include "macro_micro_controller.h"
#include "micro_arm.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class Micromanipulator;
}
QT_END_NAMESPACE

class Micromanipulator : public QMainWindow
{
    Q_OBJECT

    // 在线修正可选算法
    enum class OnlineUpdateMethod
    {
        Broyden = 0,               ///< 传统的一阶 Broyden 更新。
        SlidingWindowLeastSquares, ///< 窗口化的批量最小二乘（SWLS）。
        RecursiveLeastSquaresFF    ///< 带遗忘因子的递归最小二乘（RLS-FF）。
    };

public:
    Micromanipulator(QWidget *parent = nullptr);
    ~Micromanipulator();

private:
    Ui::Micromanipulator *ui;
    QButtonGroup *m_displayModeGroup = nullptr; ///< 单选按钮分组，确保图像显示模式互斥。
    enum class ClickDriveMode
    {
        TrackTipSingleClick = 0, ///< 传统模式：以实时识别的针尖像素为起点，单击目标即驱动。
        TwoClickDelta = 1        ///< 新模式：两次点击分别给出起点/终点，再换算驱动。
    };

private slots:
    void on_BtnCameraOnOff_clicked();

    void ShowCamera();

    // void bindOpenCVWindowToQt();

    // void displayProcessedImage(const cv::Mat &processedImage);  // 接收处理后的图像并显示

    // void captureAndProcess();  // 定时器触发槽函数，用于捕获并传递图像

    void updateResolutionAndFrameRates();

    void on_BtnSetResolutionAndFrameRate_clicked();

    void on_BtnGetCurrentPosition_clicked();

    void on_BtnMove_clicked();

    void on_BtnSetVR_clicked();

    void on_BtnSerialPortOnOff_clicked();

    void on_BtnSerialPortSend_clicked();

    void SerialReadData(const QByteArray &rawBytes, const QString &hexView);
    void handleMacroArmFeedback(const QString &text); // 统一处理宏观机械臂反馈，缓存原始 JSON 文本并刷新 UI。

    void updateLineEdit(int value);

    void updateSliderFromText();

    void on_BtnGetTrans_clicked();

    void on_rBtnRelativeMove_clicked();

    void on_rBtnAbsoluteMove_clicked();

    void on_BtnSetOrigin_clicked();

    void on_BtnInterruptMove_clicked();

    void on_BtnAngleTrans_clicked();

    void on_BtnGetSingleLoopAngle_clicked();

    void on_BtnInterrupt_clicked();

    void on_BtnGetMultiLoopAngle_clicked();

    void on_BtnMoveToZero_clicked();

    void on_BtnGetTipLength_clicked();

    void on_BtnGetParameter_u_clicked();

    void on_BtnGetTipLength_2_clicked();

    void on_BtnGetTipLength_3_clicked();

    void on_SaveAnglePIDToRAM_clicked();

    void on_SaveSpeedPIDToRAM_clicked();

    void on_SavePIDToROM_clicked();

    void on_BtnGetFrame_clicked();

    void on_BtnSerialPortOnOff_2_clicked();

    void on_BtnReadCurrentAngle_clicked();

    void on_BtnRCM_clicked();

    void on_BtnSocketConnect_clicked();

    void on_BtnSendMessage_clicked();

    void on_CheckBoxEnableRobot_clicked();

    void on_BtnClearError_clicked();

    void on_BtnSetAccL_clicked();

    void on_BtnSetSpeedL_clicked();

    void on_BtnGetCurrentState_clicked();

    void on_BtnMoveJ_clicked();

    void on_BtnMoveL_clicked();

    void on_Btn_RobotStop_clicked();

    void on_Btn_RobotSlowStop_clicked();

    void on_Btn_calibration_MicRobot_clicked();

    void on_BtnGetBackImage_clicked();

    void on_Btn_calibration_MacRobot_clicked();

    void on_BtnAutoFocus_clicked();



    void on_Btn_EyeHandCalibration_clicked();

    void on_Btn_CameraCalibration_clicked();

    void on_Btn_EyeHandCalibration_solute_clicked();

    void on_BtnRecordTipError_clicked();

public:
    //相机相关
    QTimer *timer1; // 声明QTimer对象
    cv::Mat BackImage;
    cv::Mat frame;
    cv::Mat m_lastDisplayedFrame; ///< 最近一次在界面上展示的画面，用于截图保存。
    cv::Mat H_calib;//(全局)视觉与机械臂空间映射
    cv::Mat H1;
    cv::Mat alpha_L;//偏航角补偿轨迹与α的映射

public:
    //串口相关
    bool getSerialPortConfig();         //串口初始化
    bool getSerialPortConfig_2();
    QString swapBytes_1(qint32 data);  //交换字节顺序32位
    QString swapBytes_2(quint16 data);  //交换字节顺序16位
    QString getSumFromHex(QString HexString);
    float roundToTwoDecimals(const QString& text);  //保留后两位小数
    void GetTrans(qint32 x, qint32 y);//求解变换矩阵
    void Delay(int Times_ms);


    //求解变换矩阵相关
    int VisualPosition_X;//当前针尖位置（图像计算/相机坐标）
    int VisualPosition_Y;
    int RobotPosition_X = 0;//由图像转换为实际针尖位置（机械臂坐标）
    int RobotPosition_Y = 0;
    //用于矩阵变换的三组对应坐标
    int camera_x1, camera_y1;
    int camera_x2, camera_y2;
    int camera_x3, camera_y3;
    qint32 trans_x = 10000;
    qint32 trans_y = 10000;
    float a, b, c, d, e, f = 0.0;
    bool TransMatGet = false;
    cv::Point2f m_filteredTipPixel = cv::Point2f(0.0f, 0.0f); ///< 图像处理后通过指数平滑得到的针尖像素坐标。
    bool m_hasFilteredTipPixel = false; ///< 指示当前是否存在可复用的上一帧针尖估计。
    cv::Point2f m_lastMeasuredTip = cv::Point2f(0.0f, 0.0f); ///< 最近一次原始检测到的针尖像素坐标。
    bool m_hasLastMeasuredTip = false; ///< 标记是否已经记录过上一帧测得的针尖位置。
    cv::Point2f m_lastFocusTip = cv::Point2f(0.0f, 0.0f); ///< 自动聚焦过程中最近一次可用于 ROI 的针尖位置（避免 UI 线程重置导致的丢失）。
    bool m_hasLastFocusTip = false; ///< 是否已缓存上一帧的针尖用于聚焦 ROI。
    cv::Rect m_lastFocusRoi; ///< 最近一次用于聚焦计算的 ROI，便于在检测失败时维持位置。
    bool m_hasLastFocusRoi = false; ///< 是否已有可复用的 ROI。
    bool m_lastFocusRoiAnchoredToTip = false; ///< 最近一次 ROI 是否由针尖驱动，便于可视化标注。
    cv::Point2f m_filteredFocusCenter = cv::Point2f(0.0f, 0.0f); ///< ROI 中心的平滑结果，用于抑制抖动。
    cv::Point2f m_lastRawFocusCenter = cv::Point2f(0.0f, 0.0f); ///< ROI 平滑前的原始中心，便于日志与可视化。
    bool m_hasFilteredFocusCenter = false; ///< 是否已有平滑后的 ROI 中心。
    double m_focusRoiSmoothingAlpha = 0.32; ///< ROI 中心指数平滑系数（越大越追随实时检测）。
    double m_focusRoiMaxJumpPixels = 24.0; ///< ROI 单步允许的最大跳变距离，过大视为异常抖动并被限幅。
    double m_tipSmoothingAlpha = 0.55; ///< 全局固定平滑系数（无自适应模式下使用）。
    double m_tipSmoothingAlphaStatic = 0.15; ///< 静止/微动状态下使用的平滑系数，强调稳定性。
    double m_tipSmoothingAlphaMoving = 0.85; ///< 明显运动状态下使用的平滑系数，强调响应速度。
    double m_tipMotionThresholdPixels = 15.0; ///< 判定针尖进入运动状态的像素阈值。
    cv::Point2f m_tipAxisDirection = cv::Point2f(1.0f, 0.0f); ///< 针体主轴的平滑方向，用于保持吸持针末端稳定。
    bool m_hasTipAxisDirection = false; ///< 标记是否已有历史主轴可供平滑参考。
    double m_tipAxisSmoothingAlpha = 0.55; ///< 主轴方向指数平滑系数，兼顾稳定性与实时性的折中值。
    double m_tipAxialBackstopPixels = 18.0; ///< 忽略基准点反向的像素阈值，防止噪声点破坏末端拟合。
    //用于旋转求解的参数
    double Parameter_u = 0.0;//比例系数u
    int x0,x1,x2;//三组x值
    int y0, y1;//修改后的x0y0，x1y1
    double CosAlpha = cos(2 * M_PI / 180.0);
    double SinAlpha = sin(2 * M_PI / 180.0);
    double Ltip = 0.0;//针长
    double CurrentAngle;//当前俯仰轴角度值

    cv::Mat Matrix_Camera_Mac;
    cv::Mat Matrix_Camera_Mic;
    QString MicInfo;
    QByteArray m_pendingMicPositionBytes;   // 累积“获取当前位置”指令的返回数据（原始字节序列）。
    bool m_waitingForMicPositionResponse = false; // 标志位：当前是否在等待微动机械臂回复坐标。
    void handleMicPositionFrame(const QByteArray &frameBytes); // 解析微动机械臂返回的 13 字节坐标数据帧并更新内部状态。
    QString bytesToUpperHex(const QByteArray &bytes) const; // 将原始字节序列转换为紧凑的大写十六进制字符串。
    QString formatHexForDisplay(const QString &compactHex) const; // 将紧凑十六进制串格式化为带空格的可读形式。

private:
    cv::Point2f adaptiveSmoothTip(const cv::Point2f &measuredTip, bool hasMeasurement);
    //串口参数
    bool mIsOpen;
    QString mPortName;
    QString mBaudRate;
    QString mParity;
    QString mDataBits;
    QString mStopBits;
    QString accumulatedText;// 累积数据
    bool mIsOpen_2;
    QString mPortName_2;
    QString mBaudRate_2;
    // ImageProcessingThread *processingThread; // 处理线程对象

    // TCP通信对象
    QTcpSocket *m_RobotSocket;  // 8080端口
    QTcpSocket *m_feedbackSocket;     // ⚠️ 8089/8090 反馈端口暂未启用，仅保留指针占位。
    QString m_lastMacroArmFeedback;   ///< 缓存最近一次宏观机械臂返回的原始字符串，按钮查询失败时可回退使用。
    bool mSocketIsOpen = 0;
    bool mStop = 0;

    QTimer *m_microJogTimer = nullptr; ///< 按压方向键时的循环微动定时器。
    QPoint m_microJogDirection{0, 0};  ///< 当前按压的方向（X, Y，单位：符号）。
    bool m_isMicroJogging = false;     ///< 是否处于微动重复触发状态。

    QTimer *moveTimer;        // 定时器指针，用于周期触发移动
    QSet<int> pressedKeys;    // 集合，存储当前被按下的按键


public:
    void TcpSocketSend(QTcpSocket *SelectedSocket,QString Message);
    void readSocketData();
    JsonExplain *m_JsonExplain;

    //机械臂功能
public:
    void drawCornerPoints(const cv::Mat& src_img,
                          const std::vector<std::vector<cv::Point2f>>& img_points);

    cv::Mat cameraMatrix; // 相机内参矩阵
    cv::Mat distCoeffs;// 畸变系数

    // ====== 存储全局容器 ======
    std::vector<cv::Mat> Rbase2end_list, tbase2end_list;
    std::vector<cv::Mat> Rboard2camera_list, tboard2camera_list;

    cv::Mat R_cam2base, t_cam2base;

    int curX = 0, curY = 0, curZ = 0;
    int curRx = 0, curRy = 0, curRz = 0;
    int mic_X = 0, mic_Y = 0, mic_Z = 0;
    bool eventFilter(QObject *obj, QEvent *event);
    int clicked_imgX, clicked_imgY;
    ClickDriveMode m_clickDriveMode = ClickDriveMode::TrackTipSingleClick; ///< 点击驱动模式。
    bool m_pendingTwoClickStart = false; ///< 双击模式下是否已记录首点、等待第二次点击。
    cv::Point2i m_twoClickStartPixel = cv::Point2i(0, 0); ///< 双击模式首点（起始针尖像素）。
    cv::Point2i m_lastTwoClickTargetPixel = cv::Point2i(0, 0); ///< 上一次双击模式的终点像素（k次目标）。
    bool m_hasLastTwoClickTargetPixel = false; ///< 是否已记录上一次双击终点像素。
    bool m_lastCommandFromTwoClickMode = false; ///< 上一次驱动命令是否来自双击模式。
    double m_twoClickDeltaMinPixels = 1.5; ///< 双击修正样本的最小像素阈值，抑制微小点选抖动。
    double m_twoClickResidualClampPixels = 4.0; ///< 双击测量与预测不一致时的残差限幅（像素）。
    double m_twoClickUsePredictedBlend = 0.35; ///< 双击样本融合时预测像素位移的权重[0,1]。

    double  alpha_calib_rad;  // 标定时的 RZ 角（弧度）


    // 类成员
    cv::Matx22d H1_linear;              // H1 的前 2x2 部分
    cv::Vec2d last_delta_arm;           // 上一次发给机械臂的差值
    cv::Point2d last_visual;            // 上一次视觉像素点
    bool has_last_sample = false;
    // ====== 在线修正参数（集中声明，便于 UI 配置与说明） ======
    double lambda_broyden = 1e-5; ///< Broyden 正则化项，防止奇异（λ）。
    double gamma_broyden  = 0.85; ///< Broyden 更新步长系数（γ）。
    int    swls_window_size = 10; ///< SWLS 采样窗口大小（最近 N 条样本）。
    double swls_ridge       = 1e-6; ///< SWLS 岭回归系数，提升数值稳定性。
    double rls_forgetting_factor = 0.98; ///< RLS-FF 遗忘因子（λ），越小越重视最新样本。
    double rls_initial_cov       = 1000.0; ///< RLS-FF 协方差初始化量级，越大越易快速自适应。
    OnlineUpdateMethod online_update_method = OnlineUpdateMethod::Broyden; ///< 当前在线修正算法。
    std::deque<std::pair<cv::Vec2d, cv::Vec2d>> online_samples; ///< 存储最近的 (Δpixel, Δarm) 样本。
    cv::Matx<double,4,1> rls_theta = cv::Matx<double,4,1>::zeros(); ///< RLS-FF 参数向量（展平的 2×2 矩阵）。
    cv::Matx44d rls_cov = cv::Matx44d::eye(); ///< RLS-FF 协方差矩阵。
    bool rls_initialized = false; ///< RLS-FF 是否已经完成初始化。



protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override   // 重写键盘按下事件
    {
        int key = event->key();                     // 获取当前按下的键值
        if (!pressedKeys.contains(key)) {           // 如果集合里没有这个键
            pressedKeys.insert(key);                // 记录该按键到 pressedKeys 集合
        }

        if (!moveTimer->isActive()) {               // 如果定时器还没启动
            moveTimer->start();                     // 启动定时器（开始循环调用）
        }

        QMainWindow::keyPressEvent(event);          // 调用父类，保证其他功能正常
    }

    void keyReleaseEvent(QKeyEvent *event) override // 重写键盘释放事件
    {
        int key = event->key();                     // 获取当前释放的键值
        if (pressedKeys.contains(key)) {            // 如果集合里有这个键
            pressedKeys.remove(key);                // 从集合中移除该按键
        }

        if (pressedKeys.isEmpty() && moveTimer->isActive()) {
            // 如果集合为空（没有按下的键）并且定时器还在运行
            moveTimer->stop();                      // 停止定时器
        }

        QMainWindow::keyReleaseEvent(event);        // 调用父类，保证其他功能正常
    }


private slots:
    // void processMovement()    // 定时器触发时，检查哪些键被按下
    // {

    //     int step = 50; // 每次位姿的增量步长，可调大调小// Z方向：Q (-Z)，E (+Z)
    //     if (pressedKeys.contains(Qt::Key_W)) curZ += step; // W 键 → Z 正方向移动
    //     if (pressedKeys.contains(Qt::Key_S)) curZ -= step; // S 键 → Z 负方向移动

    //     // 方向键 ↑ ↓ ← → 作为xy控制
    //     if (pressedKeys.contains(Qt::Key_8)) curY -= step; // ↑ 键 → Y 负方向移动
    //     if (pressedKeys.contains(Qt::Key_2)) curY += step; // ↓ 键 → Y 正方向移动
    //     if (pressedKeys.contains(Qt::Key_4)) curX += step; // ← 键 → X 正方向移动
    //     if (pressedKeys.contains(Qt::Key_6)) curX -= step; // → 键 → X 负方向移动
    //     movep_canfd(curX, curY, curZ, 3142, 0, -1571, 1, 2, 800);
    // }

    void on_Btn_KeepCenter_clicked();
    void on_BtnMoveC_clicked();
    void on_BtnMoveVia_clicked();
    void on_BtnMoveToPixel_clicked();
    void on_comboClickDriveMode_currentIndexChanged(int index);

    void handleCameraOpened();
    void handleCameraClosed();
    void handleCameraFrame(const cv::Mat &frame);
    void handleProcessedImage(const ImageProcessor::ProcessedImage &result);
    void on_BtnRecordVideo_clicked();
    void on_cBoxSmoothingMode_currentIndexChanged(int index);

private:
    void triggerMicroArmMoveForPixel(int pixelX, int pixelY);
    void triggerMicroArmMoveByPixels(const cv::Point2i &sourcePixel,
                                     const cv::Point2i &targetPixel,
                                     bool useTargetForDisplay = true,
                                     bool isTwoClickCommand = false);
    ImageProcessor::ProcessedImage runImageProcessingPipeline(const ImageProcessor::FrameRequest &request);
    bool startRecording(const cv::Size &frameSize, double fps);
    void stopRecording();
    void resetTipSmoothingState();
    cv::Point2f smoothTip(const cv::Point2f &measuredTip, bool hasMeasurement);
    cv::Point2f kalmanPredictiveSmooth(const cv::Point2f &measuredTip, bool hasMeasurement);
    cv::Rect buildFocusRoi(const cv::Mat &bgrFrame, cv::Mat &roiOut,
                           cv::Point2f &roiCenter, bool &roiAnchoredToTip,
                           cv::Point2f &rawRoiCenter);
    QString focusRoiSourceToString(bool roiAnchoredToTip) const;
    cv::Mat normalizeTo8U(const cv::Mat &gray) const;

    struct FocusMeasureSample
    {
        int stepIndex = 0;
        int targetZ = 0;
        int roiX = 0;
        int roiY = 0;
        int roiW = 0;
        int roiH = 0;
        double roiCenterRawX = 0.0; ///< ROI 平滑前的中心 x。
        double roiCenterRawY = 0.0; ///< ROI 平滑前的中心 y。
        double roiCenterX = 0.0;
        double roiCenterY = 0.0;
        bool roiAnchoredToTip = false;
        double variance = 0.0;
        double tenengrad = 0.0;   ///< Tenenbaum 梯度能量（常用于显微图像聚焦）。
        double brenner = 0.0;     ///< Brenner 梯度，经典的聚焦对比指标。
        double laplacianVar = 0.0;///< Laplacian 方差，常用锐度度量，附加参考。
        double entropy = 0.0;     ///< 灰度熵，反映纹理/对比度复杂度，附加参考。
        double highFrequency = 0.0; ///< 高频能量指标，衡量纹理细节丰富度。
    };

    FocusMeasureSample evaluateFocusMeasures(const cv::Mat &bgrFrame, int stepIndex, int targetZ) const;
    double focusVariance(const cv::Mat &gray) const;
    double focusTenengrad(const cv::Mat &gray) const;
    double focusBrenner(const cv::Mat &gray) const;
    double focusLaplacianVariance(const cv::Mat &gray) const;
    double focusEntropy(const cv::Mat &gray) const;
    double focusHighFrequencyEnergy(const cv::Mat &gray) const;
    QString buildDefaultFocusLogPath() const;
    QString buildDefaultTipErrorLogPath() const;

    CameraModule m_cameraModule;              ///< 封装的相机模块。
    ImageProcessor m_imageProcessorModule;    ///< 图像处理线程模块。
    MicroArm m_microArmModule;                ///< 微动机械臂模块。
    MacroArm m_macroArmModule;                ///< 宏观机械臂模块。
    MacroMicroController m_macroMicroController; ///< 宏微协调控制模块。

    QSerialPort mSerialPort_2;

    enum class TipSmoothingMode
    {
        Adaptive,
        GlobalFixed,
        KalmanPredictive
    };

    TipSmoothingMode m_tipSmoothingMode = TipSmoothingMode::Adaptive; ///< 控制当前针尖平滑模式。
    cv::KalmanFilter m_tipKalmanFilter; ///< 针尖位置卡尔曼滤波器，包含速度状态用于预测。
    bool m_kalmanInitialized = false;   ///< 卡尔曼滤波器是否已按当前模式完成初始化。
    // ===== Kalman 滤波可调参数（帧为时间单位，如需调节跟随/平滑效果请优先改这里） =====
    // 调参提示：提高过程噪声 → 响应更快但更抖；提高测量噪声 → 更平滑但滞后更明显。
    float m_kalmanProcessNoisePos = 2e-3f; ///< 位置过程噪声（越大越灵敏）。
    float m_kalmanProcessNoiseVel = 5e-3f; ///< 速度过程噪声（越大越灵敏）。
    float m_kalmanMeasurementNoise = 8e-3f; ///< 测量噪声（越小越贴近观测）。
    float m_kalmanInitialError = 3.0f; ///< 初始误差协方差（越大越快速收敛到观测）。

    cv::VideoWriter m_videoWriter; ///< 录屏写入器，保存与原始图像分辨率一致的视频。
    bool m_isRecording = false;    ///< 录屏状态标记。
    cv::Size m_recordingFrameSize; ///< 当前录屏使用的帧尺寸，确保不经过 UI 缩放。
    double m_recordingFps = 30.0;  ///< 录屏帧率，默认使用计时器推算值。
    QString m_recordingFilePath;   ///< 当前录屏文件的完整路径。

    // ====== 图像处理线程内的缓冲资源，避免每帧重复构造 ======
    cv::Mat m_cachedUndistortMap1;       ///< 上一次生成的去畸变映射表（x/y）。
    cv::Mat m_cachedUndistortMap2;       ///< 上一次生成的去畸变映射表（像素索引）。
    cv::Size m_cachedUndistortSize;      ///< 映射表对应的帧尺寸，用于判定是否需要重建。
    cv::Mat m_cachedCameraMatrix;        ///< 生成映射表时使用的相机内参。
    cv::Mat m_cachedDistCoeffs;          ///< 生成映射表时使用的畸变系数。
    cv::Ptr<cv::CLAHE> m_clahe;          ///< 复用的 CLAHE 对象，避免每帧重新 new。

    bool ensureUndistortMaps(const ImageProcessor::FrameRequest &request,
                             cv::Mat &map1, cv::Mat &map2);
    void resetOnlineUpdateState();
    void startMicroJog(const QPoint &direction);
    void stopMicroJog();
    void triggerMicroJogStep();
    void startTipErrorRecording(const QString &filePath);
    void stopTipErrorRecording();
    void writeTipErrorSample(const ImageProcessor::ProcessedImage &result);

    bool m_isTipErrorRecording = false; ///< 针尖滤波误差记录状态。
    QFile m_tipErrorFile;              ///< 误差记录输出文件。
    QTextStream m_tipErrorStream;      ///< 误差记录输出流。
    QString m_tipErrorFilePath;        ///< 误差记录文件路径。
    qint64 m_tipErrorFrameIndex = 0;   ///< 误差记录的帧计数。
    cv::Point2f m_logAdaptiveFilteredTip = cv::Point2f(0.0f, 0.0f); ///< 自适应平滑记录用缓存。
    bool m_logAdaptiveHasFilteredTip = false; ///< 自适应平滑记录用缓存是否有效。
    cv::Point2f m_logAdaptiveLastMeasuredTip = cv::Point2f(0.0f, 0.0f); ///< 自适应平滑记录用上一帧测量值。
    bool m_logAdaptiveHasLastMeasuredTip = false; ///< 自适应平滑记录用上一帧测量值是否有效。
    cv::Point2f m_logGlobalFilteredTip = cv::Point2f(0.0f, 0.0f); ///< 全局平滑记录用缓存。
    bool m_logGlobalHasFilteredTip = false; ///< 全局平滑记录用缓存是否有效。
    cv::KalmanFilter m_logKalmanFilter; ///< 误差记录专用的卡尔曼滤波器。
    bool m_logKalmanInitialized = false; ///< 误差记录用卡尔曼滤波器是否初始化。
    cv::Point2f m_logKalmanFilteredTip = cv::Point2f(0.0f, 0.0f); ///< 误差记录用卡尔曼滤波结果。
    bool m_logKalmanHasFilteredTip = false; ///< 误差记录用卡尔曼滤波结果是否有效。
};

// 用于在线修正 H1 的线性部分
static inline cv::Matx22d updateMappingBroyden(
    const cv::Matx22d& H1_linear,
    const cv::Vec2d& delta_pixel,   // 实际像素改变量
    const cv::Vec2d& delta_arm_cmd, // 上次发给机械臂的位移
    double lambda_broyden,
    double gamma_broyden)
{
    double norm2 = delta_pixel.dot(delta_pixel);
    if (norm2 < 1e-12) return H1_linear;  // 像素改变量太小就跳过

    // 预测误差 r = 上次下发位移 - (当前H1 * 实际像素变化)
    cv::Vec2d residual = delta_arm_cmd - H1_linear * delta_pixel;

    double denom = norm2 + lambda_broyden;
    cv::Matx22d rank1(
        residual[0]*delta_pixel[0]/denom, residual[0]*delta_pixel[1]/denom,
        residual[1]*delta_pixel[0]/denom, residual[1]*delta_pixel[1]/denom
        );

    return H1_linear + gamma_broyden * rank1;
}

// 将 2×2 矩阵展平/还原，便于递推运算
static inline cv::Matx<double,4,1> flattenLinearMapping(const cv::Matx22d& H)
{
    return cv::Matx<double,4,1>(H(0,0), H(0,1), H(1,0), H(1,1));
}

static inline cv::Matx22d unflattenLinearMapping(const cv::Matx<double,4,1>& v)
{
    return cv::Matx22d(v(0,0), v(1,0), v(2,0), v(3,0));
}

// 滑动窗口最小二乘：在最近 N 条样本上批量求解 H
static inline cv::Matx22d updateMappingSwls(
    const std::deque<std::pair<cv::Vec2d, cv::Vec2d>>& samples,
    const cv::Matx22d& fallback,
    double ridge)
{
    const int N = static_cast<int>(samples.size());
    if (N < 2) return fallback; // 样本太少时保持原值

    cv::Mat1d A(2 * N, 4), b(2 * N, 1);
    for (int i = 0; i < N; ++i) {
        const cv::Vec2d& dx = samples[i].first;  // 像素变化
        const cv::Vec2d& dy = samples[i].second; // 机械臂指令
        const int row = 2 * i;
        A(row, 0) = dx[0]; A(row, 1) = dx[1]; A(row, 2) = 0.0; A(row, 3) = 0.0; b(row, 0) = dy[0];
        A(row + 1, 0) = 0.0; A(row + 1, 1) = 0.0; A(row + 1, 2) = dx[0]; A(row + 1, 3) = dx[1]; b(row + 1, 0) = dy[1];
    }

    cv::Mat1d AtA = A.t() * A;
    if (ridge > 0) {
        AtA += ridge * cv::Mat1d::eye(4, 4);
    }
    cv::Mat1d Atb = A.t() * b;

    cv::Mat1d h;
    if (!cv::solve(AtA, Atb, h, cv::DECOMP_SVD)) {
        return fallback;
    }
    return cv::Matx22d(h(0, 0), h(1, 0), h(2, 0), h(3, 0));
}

// 带遗忘因子的递归最小二乘（针对 2×2 映射矩阵）
static inline cv::Matx22d updateMappingRlsFf(
    const cv::Vec2d& delta_pixel,
    const cv::Vec2d& delta_arm_cmd,
    double forgetting_factor,
    double initial_cov,
    const cv::Matx22d& current,
    cv::Matx<double,4,1>& theta,
    cv::Matx44d& covariance,
    bool& initialized)
{
    if (forgetting_factor <= 0.0 || forgetting_factor > 1.0) {
        return current; // 遗忘因子非法时直接跳过
    }

    if (!initialized) {
        theta = flattenLinearMapping(current);
        covariance = cv::Matx44d::eye() * initial_cov;
        initialized = true;
    }

    auto update_once = [&](double y, const cv::Vec4d& phi)
    {
        cv::Matx<double,4,1> phi_vec(phi[0], phi[1], phi[2], phi[3]);
        cv::Matx<double,1,4> phi_T(phi[0], phi[1], phi[2], phi[3]);
        cv::Matx<double,4,1> Pphi = covariance * phi_vec;
        double denom = forgetting_factor + (phi_T * Pphi)(0, 0);
        if (std::abs(denom) < 1e-12) return; // 避免除零

        cv::Matx<double,4,1> K = (1.0 / denom) * Pphi;
        double residual = y - (phi_T * theta)(0, 0);
        theta = theta + K * residual;
        covariance = (covariance - K * (phi_T * covariance)) * (1.0 / forgetting_factor);
    };

    update_once(delta_arm_cmd[0], cv::Vec4d(delta_pixel[0], delta_pixel[1], 0.0, 0.0));
    update_once(delta_arm_cmd[1], cv::Vec4d(0.0, 0.0, delta_pixel[0], delta_pixel[1]));

    return unflattenLinearMapping(theta);
}



inline void m_fitCircleKasa(const std::vector<cv::Point2d>& P,
                          double& cx, double& cy, double& r)
{
    const int N = (int)P.size();                                      // 样本数
    cv::Mat1d A(N, 3), b(N, 1);                                       // 线性方程 A*[a b c]^T = b
    for (int i = 0; i < N; ++i) {                                     // 逐点填充
        const double x = P[i].x, y = P[i].y;                          // 取坐标
        A(i,0) = x;                                                   // A 第一列 = x
        A(i,1) = y;                                                   // A 第二列 = y
        A(i,2) = 1.0;                                                 // A 第三列 = 1
        b(i,0) = -(x*x + y*y);                                        // 右端项 = -(x^2+y^2)
    }
    cv::Mat1d abc;                                                    // 待求 a,b,c
    cv::solve(A, b, abc, cv::DECOMP_SVD);                             // 最小二乘解
    const double a = abc(0,0), bcoef = abc(1,0), c = abc(2,0);        // 解出 a,b,c
    cx = -a / 2.0;                                                    // 圆心 x
    cy = -bcoef / 2.0;                                                // 圆心 y
    r  = std::sqrt(std::max(0.0, cx*cx + cy*cy - c));                 // 半径 r（数值安全）
}

//================= 2) 用 α 对齐相位：估计 φ 与 方向 s∈{+1,-1} =================//
// 思路：对每个点算几何角 θ_i=atan2(y-cy, x-cx)。希望 θ_i ≈ φ + s*α_i（s=±1）。
// 用复数平均：z_i = exp(j*(θ_i - s*α_i))，其平均的相位就是 φ（取 s 使残差更小）。
inline void m_fitPhaseWithAlpha(const std::vector<cv::Point2d>& P,
                              const std::vector<double>& A,
                              double cx, double cy,
                              int& s, double& phi, double& rmse_rad)
{
    auto fit_for_s = [&](int sgn, double& phi_out, double& rmse_out)  // 内部：给定 sgn 求 φ 和误差
    {
        double csum = 0.0, ssum = 0.0;                                // 累加 cos/sin
        const int N = (int)P.size();                                  // 点数
        std::vector<double> err(N);                                    // 角误差（弧度）
        for (int i = 0; i < N; ++i) {                                  // 逐点
            const double theta = std::atan2(P[i].y - cy, P[i].x - cx); // 几何角 θ_i
            const double ang   = theta - sgn * A[i];                   // θ_i - s*α_i
            csum += std::cos(ang);                                     // 累加 cos
            ssum += std::sin(ang);                                     // 累加 sin
        }
        phi_out = std::atan2(ssum, csum);                              // φ = arg(平均复数)
        // 计算 RMSE（弧度）：err_i = wrap(θ_i - (φ + s*α_i))
        double se = 0.0;                                               // 误差平方和
        for (int i = 0; i < (int)P.size(); ++i) {
            const double theta = std::atan2(P[i].y - cy, P[i].x - cx); // θ_i
            double diff = theta - (phi_out + sgn * A[i]);              // 角差
            // wrap 到 [-pi,pi]
            diff = std::atan2(std::sin(diff), std::cos(diff));         // 归一化角差
            se += diff * diff;                                         // 累加平方
        }
        rmse_out = std::sqrt(se / P.size());                           // RMSE（弧度）
    };

    double phi_p, rmse_p; fit_for_s(+1, phi_p, rmse_p);                // 尝试 s=+1
    double phi_m, rmse_m; fit_for_s(-1, phi_m, rmse_m);                // 尝试 s=-1
    if (rmse_p <= rmse_m) { s = +1; phi = phi_p; rmse_rad = rmse_p; }  // 选更小误差的 s 和 φ
    else                     { s = -1; phi = phi_m; rmse_rad = rmse_m; }
}

//================= 3) 根据 (cx,cy,r,φ,s) 构造 L，z 行按你的需求 =================//
// L = [ cx, r*cosφ, -r*sinφ;
//       cy, r*sinφ,  r*cosφ;
//       z0,   0    ,   0   ]
inline cv::Mat m_buildL(double cx, double cy, double r, double phi, double z0)
{
    const double c = std::cos(phi);                                    // cosφ
    const double s = std::sin(phi);                                    // sinφ
    cv::Mat L = (cv::Mat_<double>(3,3)
                     << cx,  r*c, -r*s,                                             // 第一行
                 cy,  r*s,  r*c,                                             // 第二行
                 z0,  0.0,  0.0);                                            // 第三行（z 恒定）
    return L;                                                           // 返回 L
}

//================= 4) 一键接口：给 (x,y,z,α) 点，返回 L =================//
// 过程：Kåsa 拟合圆 → 用 α 对齐 φ 与方向 → 构造 L。
inline cv::Mat m_fitCircle2D_geometry_then_alpha(const std::vector<cv::Vec4d>& pts,
                                        double* cx=nullptr, double* cy=nullptr,
                                        double* r=nullptr,  double* phi=nullptr,
                                        int* sgn_out=nullptr, double* rmse_alpha=nullptr,
                                        bool print=true)
{
    const int N = (int)pts.size();                                     // 样本数
    if (N < 3) { if (print) std::cout << "需要 ≥3 点\n"; return cv::Mat(); }

    // 拆分 (x,y) 与 α，z0 你可以用首点或均值；我这里用“均值”，更符合你给的样例
    std::vector<cv::Point2d> P(N); std::vector<double> A(N);           // 点和角
    double z0 = 0.0;                                                   // z 的均值
    for (int i = 0; i < N; ++i) {
        P[i] = { pts[i][0], pts[i][1] };                               // 取 (x,y)
        A[i] =   pts[i][3];                                            // 取 α（弧度）
        z0  +=   pts[i][2];                                            // 累加 z
    }
    z0 /= N;                                                           // z 的平均值

    // 1) 仅用几何拟合圆（不看 α）
    double Cx, Cy, R;                                                  // 圆心与半径
    m_fitCircleKasa(P, Cx, Cy, R);                                       // Kåsa 线性解

    // 2) 用 α 拟合相位 φ 和方向 s
    int s = +1; double Phi = 0.0, rmseRad = 0.0;                       // s ∈ {+1,-1}
    m_fitPhaseWithAlpha(P, A, Cx, Cy, s, Phi, rmseRad);                  // 对齐 φ 与方向

    // 3) 构造 L（把 s 合并进 φ：若 s=-1，相当于用 φ' 且把 α 改为 -α；等价于换 φ→-φ 也能处理
    // 这里我们不改 α 的使用方式，直接把 “R(φ) · [cosα; sinα]” 写进 L 即可（φ 已考虑 s）。
    // 注：m_fitPhaseWithAlpha 已选择了更优的 s 与 φ，你在预测时仍用原 α。
    cv::Mat L = m_buildL(Cx, Cy, R, Phi, z0);                            // 拼 L 矩阵

    // 输出可选参数
    if (cx) *cx = Cx; if (cy) *cy = Cy; if (r) *r = R; if (phi) *phi = Phi; if (sgn_out) *sgn_out = s;
    if (rmse_alpha) *rmse_alpha = rmseRad;

    if (print) {
        // 简要质量指标（半径一致性）
        double meanR = 0, varR = 0;
        for (int i = 0; i < N; ++i) {
            const double dx = P[i].x - Cx, dy = P[i].y - Cy;           // 各点到圆心向量
            const double ri = std::sqrt(dx*dx + dy*dy);                // 半径样本
            meanR += ri;                                               // 累加
        }
        meanR /= N;
        for (int i = 0; i < N; ++i) {
            const double dx = P[i].x - Cx, dy = P[i].y - Cy;
            const double ri = std::sqrt(dx*dx + dy*dy);
            varR += (ri - meanR)*(ri - meanR);
        }
        varR /= std::max(1, N-1);
        const double stdR = std::sqrt(varR);

        std::cout << "圆心(cx,cy) = " << Cx << ", " << Cy << "\n";
        std::cout << "半径 r      = " << R  << "    (半径std=" << stdR << ")\n";
        std::cout << "相位 φ(rad)= " << Phi << "    方向s=" << s << "    α配准RMSE=" << rmseRad << " rad\n";
        std::cout << "L(3x3):\n"
                  << L.at<double>(0,0) << " " << L.at<double>(0,1) << " " << L.at<double>(0,2) << "\n"
                  << L.at<double>(1,0) << " " << L.at<double>(1,1) << " " << L.at<double>(1,2) << "\n"
                  << L.at<double>(2,0) << " " << L.at<double>(2,1) << " " << L.at<double>(2,2) << "\n";
    }
    return L;                                                           // 返回 L
}

static inline cv::Mat rot2d_h(double theta){
    const double c = std::cos(theta), s = std::sin(theta);
    return (cv::Mat_<double>(3,3) <<  c, -s, 0,
            s,  c, 0,
            0,  0, 1);
}


#endif // MICROMANIPULATOR_H
