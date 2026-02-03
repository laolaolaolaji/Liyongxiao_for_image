#include "micromanipulator.h"
#include "ui_micromanipulator.h"
#include <algorithm> // std::max/std::min 用于针尖投影与串口解析的范围控制
#include <cmath>
#include <fstream>  // 写文件
#include <iomanip>  // 控制小数位
#include <limits>   // 针尖检测时用于初始化极值
#include <numeric>  // std::accumulate 计算交点均值

#include <QStandardPaths>
#include <QCoreApplication>
#include <QEventLoop>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QDateTime>
#include <QFileDialog>

#include <opencv2/core/ocl.hpp>
#include <opencv2/video/tracking.hpp>


Micromanipulator::Micromanipulator(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::Micromanipulator)
    , timer1(new QTimer(this))// 初始化QTimer对象
    , m_JsonExplain(new JsonExplain(this))
    , m_cameraModule(this)
    , m_imageProcessorModule(this)
    , m_microArmModule(this)
    , m_macroArmModule(this)
    , m_macroMicroController(this)
    // ,processingThread(new ImageProcessingThread(this))
{
    ui->setupUi(this);
    ui->CameraShow->installEventFilter(this);

    // ====== 配置相机图像显示模式的单选按钮 ======
    m_displayModeGroup = new QButtonGroup(this);
    m_displayModeGroup->setExclusive(true); // 明确指定只允许单选，避免 UI 误触导致多选。
    m_displayModeGroup->addButton(ui->rBtnOriginal);
    m_displayModeGroup->addButton(ui->rBtnGrayscale);
    m_displayModeGroup->addButton(ui->rBtnClaheGray);
    m_displayModeGroup->addButton(ui->rBtnBinary);
    m_displayModeGroup->addButton(ui->rBtnEdges);
    ui->rBtnOriginal->setChecked(true); // 默认展示原始画面。

    ui->BtnRecordVideo->setText(QStringLiteral("开始录屏"));
    ui->BtnRecordTipError->setText(QStringLiteral("开始记录误差"));
    if (ui->lineFocusOutputPath->text().isEmpty()) {
        ui->lineFocusOutputPath->setText(buildDefaultFocusLogPath());
    }
    if (ui->cBoxSmoothingMode->count() == 0) {
        ui->cBoxSmoothingMode->addItem(QStringLiteral("自适应平滑"));
        ui->cBoxSmoothingMode->addItem(QStringLiteral("全局固定平滑"));
    ui->cBoxSmoothingMode->addItem(QStringLiteral("卡尔曼滤波"));
    }
    ui->cBoxSmoothingMode->setCurrentIndex(static_cast<int>(m_tipSmoothingMode));

    // 在线矩阵修正算法选择与参数绑定
    if (ui->comboOnlineUpdateMethod->count() == 0) {
        ui->comboOnlineUpdateMethod->addItem(QStringLiteral("Broyden 一阶更新"));
        ui->comboOnlineUpdateMethod->addItem(QStringLiteral("SWLS 窗口最小二乘"));
        ui->comboOnlineUpdateMethod->addItem(QStringLiteral("RLS-FF 带遗忘因子"));
    }
    ui->comboOnlineUpdateMethod->setCurrentIndex(static_cast<int>(online_update_method));
    ui->doubleSpinLambdaBroyden->setValue(lambda_broyden);
    ui->doubleSpinGammaBroyden->setValue(gamma_broyden);
    ui->spinSwlsWindowSize->setValue(swls_window_size);
    ui->doubleSpinSwlsRidge->setValue(swls_ridge);
    ui->doubleSpinRlsFf->setValue(rls_forgetting_factor);
    ui->doubleSpinRlsInitCov->setValue(rls_initial_cov);

    connect(ui->comboOnlineUpdateMethod, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int idx){ online_update_method = static_cast<OnlineUpdateMethod>(idx); resetOnlineUpdateState(); });
    connect(ui->doubleSpinLambdaBroyden, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v){ lambda_broyden = v; });
    connect(ui->doubleSpinGammaBroyden, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v){ gamma_broyden = v; });
    connect(ui->spinSwlsWindowSize, qOverload<int>(&QSpinBox::valueChanged),
            this, [this](int v){ swls_window_size = v; });
    connect(ui->doubleSpinSwlsRidge, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v){ swls_ridge = v; });
    connect(ui->doubleSpinRlsFf, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v){ rls_forgetting_factor = v; });
    connect(ui->doubleSpinRlsInitCov, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [this](double v){ rls_initial_cov = v; });

    // ====== 构建模块之间的依赖关系 ======
    m_macroMicroController.attachMicroArm(&m_microArmModule);
    m_macroMicroController.attachMacroArm(&m_macroArmModule);
    m_macroMicroController.attachCamera(&m_cameraModule);
    m_macroMicroController.attachImageProcessor(&m_imageProcessorModule);
    m_macroMicroController.attachPitchSerialPort(&mSerialPort_2);

    // UI 主线程主动拉取帧，关闭未使用的信号广播以避免额外深拷贝。
    m_cameraModule.setFrameSignalEnabled(false);

    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
    }

    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<ImageProcessor::ProcessedImage>("ImageProcessor::ProcessedImage");

    // 避免在每帧刷新时重复设置，提前锁定 QLabel 的缩放属性。
    ui->CameraShow->setScaledContents(true);

    m_imageProcessorModule.setProcessor([this](const ImageProcessor::FrameRequest &request) {
        return runImageProcessingPipeline(request);
    });

    connect(&m_imageProcessorModule, &ImageProcessor::processedFrame,
            this, &Micromanipulator::handleProcessedImage, Qt::QueuedConnection);
    m_imageProcessorModule.start();
    m_imageProcessorModule.setPriority(QThread::HighPriority);

    // 监听相机模块的状态变化，便于更新 UI。frameCaptured 信号暂由 ShowCamera 主动拉取方式取代。
    connect(&m_cameraModule, &CameraModule::cameraOpened,
            this, &Micromanipulator::handleCameraOpened);
    connect(&m_cameraModule, &CameraModule::cameraClosed,
            this, &Micromanipulator::handleCameraClosed);
    connect(&m_cameraModule, &CameraModule::frameCaptured,
            this, &Micromanipulator::handleCameraFrame);

    connect(&m_macroArmModule, &MacroArm::feedbackReceived,
            this, &Micromanipulator::handleMacroArmFeedback);



    // 你的 5 个点（弧度从 -2 到 -1）
    std::vector<cv::Vec4d> pts;
    pts.emplace_back(-327974, -35530, 156411, -2.00);
    pts.emplace_back(-286014, -22750, 156411, -1.75);
    pts.emplace_back(-248000,      0, 156411, -1.50);
    pts.emplace_back(-216889,  31534, 156411, -1.25);
    pts.emplace_back(-193813,  70558, 156411, -1.00);



    // 一键拟合（几何圆 + α相位对齐），得到 L
    double cx, cy, r, phi, rmseAlpha; int sgn;
    alpha_L = m_macroMicroController.fitYawCircleMapping(pts, &cx, &cy, &r, &phi, &sgn, &rmseAlpha, true);

    // 用 L 预测 α=-1.77 的点：xyz = L · [1,cosα,sinα]^T
    const double alpha_pred = -1.77;
    cv::Mat v = (cv::Mat_<double>(3,1) << 1.0, std::cos(alpha_pred), std::sin(alpha_pred));
    cv::Mat xyz = alpha_L * v;
    qDebug() << "预测 α=" << alpha_pred << " -> x,y,z = "
              << xyz.at<double>(0) << ", "
              << xyz.at<double>(1) << ", "
              << xyz.at<double>(2) << "\n";








    //将可用相机写入下拉列表
    const auto cameras = QMediaDevices::videoInputs();
    for (const QCameraDevice &cameraDevice : cameras)
    {
        ui->CboxCameraDevices->addItem(cameraDevice.description());
    }

    ui->BtnCameraOnOff->setText("打开相机");

    //初始化串口
    mIsOpen = false;
    ui->BtnSerialPortSend->setEnabled(false);
    ui->BtnGetCurrentPosition->setEnabled(false);
    ui->BtnMove->setEnabled(false);
    ui->BtnSetVR->setEnabled(false);
    ui->rBtnAbsoluteMove->setEnabled(false);
    ui->rBtnRelativeMove->setEnabled(false);
    ui->BtnInterruptMove->setEnabled(false);
    ui->BtnSetOrigin->setEnabled(false);
    ui->BtnGetTrans->setEnabled(false);
    //实时获取当前系统的有效串口号
    QList<QSerialPortInfo> SerialPortInfo = QSerialPortInfo::availablePorts();
    int count = SerialPortInfo.count();
    for(int i = 0; i < count; i++)
    {
        ui->CboxSerialPort->addItem(SerialPortInfo.at(i).portName());

    }
    //实时获取当前系统的有效串口号
    QList<QSerialPortInfo> SerialPortInfo_2 = QSerialPortInfo::availablePorts();
    int count2 = SerialPortInfo_2.count();
    for(int i = 0; i < count2; i++)
    {
        ui->CboxSerialPort_2->addItem(SerialPortInfo_2.at(i).portName());

    }

    //初始化网口
    m_RobotSocket = new QTcpSocket(this);
    // ====== ⚠️ 反馈端口暂不启用：保留 UI 输入但停用 8089 Socket ======
    // m_feedbackSocket = new QTcpSocket(this);
    m_feedbackSocket = nullptr; // 显式置空，防止误用未启用的反馈端口连接。

    ui->SocketPortNum1->setText("8080");
    ui->SocketPortNum2->setText("8089"); // ⚠️ 仅供界面展示，目前不建立 8089 反馈连接。
    ui->CBoxSelectSocket->clear();



    // 连接QTimer的timeout信号到自定义槽函数
    //相机计时器
    connect(timer1, &QTimer::timeout, this, &Micromanipulator::ShowCamera);
    connect(&m_microArmModule, &MicroArm::serialDataArrived,
            this, &Micromanipulator::SerialReadData);
    connect(ui->VelocitySlider,&QSlider::valueChanged,this,&Micromanipulator::updateLineEdit);
    connect(ui->SetVelocity, &QLineEdit::textChanged, this, &Micromanipulator::updateSliderFromText);

    // 获取本机IPv4地址
    foreach (const QHostAddress &address, QNetworkInterface::allAddresses()) {
        if (address.protocol()  == QHostAddress::IPv4Protocol
            && !address.isLoopback())  {
            QString ip = address.toString();
            QStringList ipParts = ip.split(".");
            if(ipParts.size()  == 4) {
                ui->IPAdress1->setText(ipParts[0]);
                ui->IPAdress2->setText(ipParts[1]);
                ui->IPAdress3->setText(ipParts[2]);
                ui->IPAdress4->setText(ipParts[3]);
                break;
            }
        }
    }
    mSocketIsOpen = 0;



    // 连接信号和槽
    connect(m_RobotSocket, &QTcpSocket::readyRead, this, &Micromanipulator::readSocketData);
    // if (m_feedbackSocket) {
    //     connect(m_feedbackSocket, &QTcpSocket::readyRead, this, &Micromanipulator::readSocketData);
    // }


    accumulatedText.clear();


    moveTimer = new QTimer(this);        // 创建定时器，绑定到 MainWindow
    moveTimer->setInterval(50);          // 设置定时器间隔为 50ms（20Hz）
    // connect(moveTimer, &QTimer::timeout, // 定时器超时时，触发 processMovement()
    //         this, &Micromanipulator::processMovement);

    m_microJogTimer = new QTimer(this);
    m_microJogTimer->setInterval(120); // 间隔留给硬件完成上一条指令
    connect(m_microJogTimer, &QTimer::timeout, this, &Micromanipulator::triggerMicroJogStep);

    auto bindMicroJogButton = [this](QPushButton *btn, const QPoint &direction) {
        if (!btn) {
            return;
        }
        connect(btn, &QPushButton::pressed, this, [this, direction]() { startMicroJog(direction); });
        connect(btn, &QPushButton::released, this, &Micromanipulator::stopMicroJog);
        connect(btn, &QPushButton::released, m_microJogTimer, &QTimer::stop);
    };

    bindMicroJogButton(ui->BtnMicroJogUp, QPoint(0, -1));
    bindMicroJogButton(ui->BtnMicroJogDown, QPoint(0, 1));
    bindMicroJogButton(ui->BtnMicroJogLeft, QPoint(1, 0));
    bindMicroJogButton(ui->BtnMicroJogRight, QPoint(-1, 0));

    ui->BtnMicroJogUp->setEnabled(false);
    ui->BtnMicroJogDown->setEnabled(false);
    ui->BtnMicroJogLeft->setEnabled(false);
    ui->BtnMicroJogRight->setEnabled(false);
    ui->spinBoxMicroJogStep->setEnabled(false);
}

Micromanipulator::~Micromanipulator()
{
    stopTipErrorRecording();
    delete ui;
}

void Micromanipulator::startMicroJog(const QPoint &direction)
{
    if (!mIsOpen) {
        ui->FeedBack->setText(tr("请先连接微动机械臂串口。"));
        stopMicroJog();
        return;
    }

    if (direction.isNull() || !m_microJogTimer) {
        return;
    }

    m_microJogDirection = direction;
    m_isMicroJogging = true;

    triggerMicroJogStep(); // 立即触发一次，减少按键延迟
    if (!m_microJogTimer->isActive()) {
        m_microJogTimer->start();
    }
}

void Micromanipulator::stopMicroJog()
{
    m_isMicroJogging = false;
    m_microJogDirection = QPoint(0, 0);
    if (m_microJogTimer) {
        m_microJogTimer->stop();
    }
}

void Micromanipulator::triggerMicroJogStep()
{
    if (!m_isMicroJogging || !mIsOpen) {
        return;
    }

    const int step = ui->spinBoxMicroJogStep->value();
    if (step <= 0) {
        return;
    }

    const int deltaX = m_microJogDirection.x() * step;
    const int deltaY = m_microJogDirection.y() * step;

    if (deltaX == 0 && deltaY == 0) {
        return;
    }

    if (!m_microArmModule.moveByDelta(deltaX, deltaY, 0)) {
        ui->FeedBack->setText(tr("微动机械臂微步指令发送失败"));
        return;
    }

    const cv::Vec3i pose = m_microArmModule.currentPose();
    mic_X = pose[0];
    mic_Y = pose[1];
    mic_Z = pose[2];
}

void Micromanipulator::closeEvent(QCloseEvent *event)
{
    if (timer1 && timer1->isActive()) {
        timer1->stop();
    }

    if (m_cameraModule.isCameraOpen()) {
        m_cameraModule.closeCamera();
    }

    stopRecording();
    // saveTipLogToCsv();

    m_imageProcessorModule.requestStop();
    m_imageProcessorModule.wait();

    QMainWindow::closeEvent(event);
}

void Micromanipulator::on_BtnCameraOnOff_clicked()
{
    // 通过 CameraModule 统一管理相机生命周期，主界面仅负责 UI 状态切换。
    if (!m_cameraModule.isCameraOpen()) {
        const int deviceIndex = ui->CboxCameraDevices->currentIndex();
        const bool opened = m_cameraModule.openCamera(deviceIndex, 1280, 720);
        if (opened) {
            qDebug() << "相机打开成功";
            updateResolutionAndFrameRates();
            ui->BtnCameraOnOff->setText("关闭相机");
            timer1->start(30);
        } else {
            qWarning() << "相机打开失败";
        }
    } else {
        m_cameraModule.closeCamera();
        qDebug() << "相机已关闭";
        ui->BtnCameraOnOff->setText("打开相机");
        timer1->stop();
    }
}

void Micromanipulator::on_BtnGetBackImage_clicked(){
    if (!m_cameraModule.captureFrame(BackImage)) {
        qWarning() << "on_BtnGetBackImage_clicked -> 抓取背景图失败";
    }
}

void Micromanipulator::on_BtnAutoFocus_clicked()
{
    if (!m_cameraModule.isCameraOpen()) {
        QMessageBox::warning(this, tr("提示"), tr("请先打开相机后再执行自动聚焦。"));
        return;
    }

    if (!m_microArmModule.isSerialPortOpen()) {
        QMessageBox::warning(this, tr("提示"), tr("请先连接微动机械臂串口。"));
        return;
    }

    const int maxCoarseSteps = ui->spinFocusSteps->value();
    const int coarseStep = ui->spinFocusCoarseStepSize->value();
    const int pauseMs = ui->spinFocusDelay->value();
    int fineTolerance = std::abs(ui->spinFocusStepSize->value());

    if (coarseStep <= 0) {
        QMessageBox::warning(this, tr("提示"), tr("粗搜步长必须大于 0。"));
        return;
    }
    if (fineTolerance < 1) {
        fineTolerance = 1;
    }

    m_hasFilteredFocusCenter = false; // 新一轮扫描前清空 ROI 平滑状态。

    auto appendFocusLog = [&](const QString &phase, const FocusMeasureSample &sample) {
        if (!m_isFocusLogRecording) {
            return;
        }
        if (!m_focusLogFile.isOpen()) {
            return;
        }
        m_focusLogStream << m_focusLogIndex << ','
                         << phase << ','
                         << sample.targetZ << ','
                         << sample.roiX << ','
                         << sample.roiY << ','
                         << sample.roiW << ','
                         << sample.roiH << ','
                         << sample.roiCenterRawX << ','
                         << sample.roiCenterRawY << ','
                         << sample.roiCenterX << ','
                         << sample.roiCenterY << ','
                         << focusRoiSourceToString(sample.roiAnchoredToTip) << ','
                         << sample.tenengrad << '\n';
        ++m_focusLogIndex;
        m_focusLogStream.flush();
    };

    int currentZ = m_microArmModule.currentPose()[2];
    const int baseZ = currentZ;
    int autofocusStepIndex = 0;

    auto moveToZ = [&](int targetZ) -> bool {
        const int delta = targetZ - currentZ;
        if (delta == 0) {
            return true;
        }
        if (!m_microArmModule.moveByDelta(0, 0, delta)) {
            QMessageBox::warning(this, tr("提示"), tr("微动机械臂移动失败，自动聚焦已终止。"));
            return false;
        }
        currentZ = targetZ;
        Delay(pauseMs);
        return true;
    };

    auto captureFocusAt = [&](int targetZ, const QString &phase, FocusMeasureSample *sampleOut) -> bool {
        if (!moveToZ(targetZ)) {
            return false;
        }

        ui->labelFocusStatus->setText(tr("%1：Z=%2").arg(phase).arg(targetZ));
        QCoreApplication::processEvents(QEventLoop::AllEvents, 50);

        cv::Mat latestFrame;
        if (!m_cameraModule.captureFrame(latestFrame) || latestFrame.empty()) {
            QMessageBox::warning(this, tr("提示"), tr("无法抓取图像，自动聚焦已终止。"));
            return false;
        }

        cv::Point2f roiCenter;
        cv::Point2f roiCenterRaw;
        bool roiAnchoredToTip = false;
        cv::Mat roiFrame;
        const cv::Rect roiRect = buildFocusRoi(latestFrame, roiFrame, roiCenter, roiAnchoredToTip, roiCenterRaw);

        FocusMeasureSample sample = evaluateFocusMeasures(roiFrame.empty() ? latestFrame : roiFrame,
                                                          autofocusStepIndex,
                                                          targetZ);
        ++autofocusStepIndex;
        sample.roiX = roiRect.x;
        sample.roiY = roiRect.y;
        sample.roiW = roiRect.width;
        sample.roiH = roiRect.height;
        sample.roiCenterRawX = roiCenterRaw.x;
        sample.roiCenterRawY = roiCenterRaw.y;
        sample.roiCenterX = roiCenter.x;
        sample.roiCenterY = roiCenter.y;
        sample.roiAnchoredToTip = roiAnchoredToTip;
        appendFocusLog(phase, sample);
        if (sampleOut) {
            *sampleOut = sample;
        }
        return true;
    };

    std::vector<FocusMeasureSample> coarseTrace;
    coarseTrace.reserve(static_cast<size_t>(maxCoarseSteps));

    FocusMeasureSample baseSample;
    if (!captureFocusAt(baseZ, QStringLiteral("粗搜-起点"), &baseSample)) {
        ui->labelFocusStatus->setText(tr("采集失败"));
        return;
    }
    coarseTrace.push_back(baseSample);

    int direction = 1;
    const int directionMode = ui->comboFocusDirection->currentIndex();
    FocusMeasureSample firstSample;

    if (directionMode == 0) {
        FocusMeasureSample plusSample;
        if (!captureFocusAt(baseZ + coarseStep, QStringLiteral("粗搜-探测"), &plusSample)) {
            ui->labelFocusStatus->setText(tr("采集失败"));
            return;
        }

        if (plusSample.tenengrad > baseSample.tenengrad) {
            direction = 1;
            firstSample = plusSample;
        } else {
            if (!moveToZ(baseZ)) {
                ui->labelFocusStatus->setText(tr("采集失败"));
                return;
            }

            FocusMeasureSample minusSample;
            if (!captureFocusAt(baseZ - coarseStep, QStringLiteral("粗搜-探测"), &minusSample)) {
                ui->labelFocusStatus->setText(tr("采集失败"));
                return;
            }

            if (minusSample.tenengrad >= plusSample.tenengrad) {
                direction = -1;
                firstSample = minusSample;
            } else {
                direction = 1;
                firstSample = plusSample;
            }
        }

        const int desiredZ = baseZ + direction * coarseStep;
        if (currentZ != desiredZ && !moveToZ(desiredZ)) {
            ui->labelFocusStatus->setText(tr("采集失败"));
            return;
        }
    } else {
        direction = (directionMode == 1) ? 1 : -1;
        if (!captureFocusAt(baseZ + direction * coarseStep, QStringLiteral("粗搜"), &firstSample)) {
            ui->labelFocusStatus->setText(tr("采集失败"));
            return;
        }
    }

    coarseTrace.push_back(firstSample);

    bool intervalFound = false;
    int intervalStartZ = baseZ;
    int intervalEndZ = baseZ;
    constexpr double kDropRatio = 0.08;

    for (int i = static_cast<int>(coarseTrace.size()); i < maxCoarseSteps; ++i) {
        FocusMeasureSample sample;
        if (!captureFocusAt(currentZ + direction * coarseStep, QStringLiteral("粗搜"), &sample)) {
            ui->labelFocusStatus->setText(tr("采集失败"));
            return;
        }

        coarseTrace.push_back(sample);

        if (coarseTrace.size() < 3) {
            continue;
        }

        const auto &low1 = coarseTrace[coarseTrace.size() - 3];
        const auto &peak = coarseTrace[coarseTrace.size() - 2];
        const auto &low2 = coarseTrace[coarseTrace.size() - 1];
        const double threshold = std::max(1e-6, peak.tenengrad * kDropRatio);

        if (peak.tenengrad > low1.tenengrad
            && peak.tenengrad > low2.tenengrad
            && (peak.tenengrad - low1.tenengrad) >= threshold
            && (peak.tenengrad - low2.tenengrad) >= threshold) {
            intervalFound = true;
            intervalStartZ = low1.targetZ;
            intervalEndZ = low2.targetZ;
            break;
        }
    }

    if (!intervalFound) {
        auto bestIt = std::max_element(coarseTrace.begin(), coarseTrace.end(),
                                       [](const FocusMeasureSample &a, const FocusMeasureSample &b) {
                                           return a.tenengrad < b.tenengrad;
                                       });
        const int bestIndex = static_cast<int>(std::distance(coarseTrace.begin(), bestIt));
        const int leftIndex = std::max(0, bestIndex - 1);
        const int rightIndex = std::min(static_cast<int>(coarseTrace.size()) - 1, bestIndex + 1);

        intervalStartZ = coarseTrace[static_cast<size_t>(leftIndex)].targetZ;
        intervalEndZ = coarseTrace[static_cast<size_t>(rightIndex)].targetZ;

        if (intervalStartZ == intervalEndZ) {
            intervalStartZ = bestIt->targetZ - coarseStep;
            intervalEndZ = bestIt->targetZ + coarseStep;
        }
    }

    int minZ = std::min(intervalStartZ, intervalEndZ);
    int maxZ = std::max(intervalStartZ, intervalEndZ);

    if (maxZ - minZ < fineTolerance) {
        const int finalZ = static_cast<int>(std::round((minZ + maxZ) / 2.0));
        FocusMeasureSample finalSample;
        if (captureFocusAt(finalZ, QStringLiteral("精搜-完成"), &finalSample)) {
            ui->labelFocusStatus->setText(tr("完成：Z=%1").arg(finalZ));
        }
    } else {
        constexpr double kGoldenRatio = (std::sqrt(5.0) - 1.0) / 2.0;
        double a = static_cast<double>(minZ);
        double b = static_cast<double>(maxZ);

        auto clampZ = [&](double pos) {
            int z = static_cast<int>(std::round(pos));
            z = std::clamp(z, minZ, maxZ);
            return z;
        };

        double x1 = b - kGoldenRatio * (b - a);
        double x2 = a + kGoldenRatio * (b - a);
        int z1 = clampZ(x1);
        int z2 = clampZ(x2);
        if (z1 == z2) {
            if (z1 + 1 <= maxZ) {
                z2 = z1 + 1;
            } else if (z1 - 1 >= minZ) {
                z1 = z1 - 1;
            }
        }

        FocusMeasureSample sample1;
        FocusMeasureSample sample2;
        if (!captureFocusAt(z1, QStringLiteral("精搜"), &sample1)
            || !captureFocusAt(z2, QStringLiteral("精搜"), &sample2)) {
            ui->labelFocusStatus->setText(tr("采集失败"));
            return;
        }

        double f1 = sample1.tenengrad;
        double f2 = sample2.tenengrad;
        int iter = 0;
        const int maxFineIter = 50;

        while ((b - a) > fineTolerance && iter < maxFineIter) {
            ++iter;
            if (f1 < f2) {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + kGoldenRatio * (b - a);
                z2 = clampZ(x2);
                if (z2 == z1) {
                    z2 = std::min(maxZ, z1 + 1);
                }
                if (!captureFocusAt(z2, QStringLiteral("精搜"), &sample2)) {
                    ui->labelFocusStatus->setText(tr("采集失败"));
                    return;
                }
                f2 = sample2.tenengrad;
            } else {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - kGoldenRatio * (b - a);
                z1 = clampZ(x1);
                if (z1 == z2) {
                    z1 = std::max(minZ, z2 - 1);
                }
                if (!captureFocusAt(z1, QStringLiteral("精搜"), &sample1)) {
                    ui->labelFocusStatus->setText(tr("采集失败"));
                    return;
                }
                f1 = sample1.tenengrad;
            }
        }

        const int finalZ = static_cast<int>(std::round((a + b) / 2.0));
        FocusMeasureSample finalSample;
        if (captureFocusAt(finalZ, QStringLiteral("精搜-完成"), &finalSample)) {
            ui->labelFocusStatus->setText(tr("完成：Z=%1").arg(finalZ));
        }
    }

    if (m_isFocusLogRecording && !m_focusLogFilePath.isEmpty()) {
        ui->labelFocusStatus->setText(tr("日志记录中：%1").arg(m_focusLogFilePath));
    }
}

void Micromanipulator::on_BtnFocusLog_clicked()
{
    if (m_isFocusLogRecording) {
        const QString savedPath = m_focusLogFilePath;
        stopFocusLogRecording();
        ui->BtnFocusLog->setText(QStringLiteral("开始记录聚焦日志"));
        if (!savedPath.isEmpty()) {
            ui->labelFocusStatus->setText(tr("日志已保存：%1").arg(savedPath));
        }
        return;
    }

    const QString filePath = ui->lineFocusOutputPath->text().trimmed();
    if (startFocusLogRecording(filePath)) {
        ui->BtnFocusLog->setText(QStringLiteral("停止记录聚焦日志"));
        ui->labelFocusStatus->setText(tr("日志记录中：%1").arg(m_focusLogFilePath));
    }
}

void Micromanipulator::ShowCamera()
{

    if (!m_cameraModule.isCameraOpen()) {
        qDebug() << "Camera is not opened!";
        return; // 相机未打开，跳过处理
    }

    cv::Mat latestFrame;
    if (!m_cameraModule.captureFrame(latestFrame) || latestFrame.empty()) {
        qDebug() << "Empty frame captured!";
        return; // 跳过处理
    }

    frame = latestFrame;

    ImageProcessor::FrameRequest request;
    request.frame = latestFrame;
    request.applyCalibration = ui->cBtnCameraCalibrated->isChecked();
    if (request.applyCalibration) {
        request.cameraMatrix = cameraMatrix;
        request.distCoeffs = distCoeffs;
    }
    request.flipImage = ui->rBtnImageFlip->isChecked();
    if (!BackImage.empty()) {
        request.background = BackImage;
    }

    // 根据 UI 中的单选按钮决定要展示的图像模式，并传递给后台线程。
    if (ui->rBtnBinary->isChecked()) {
        request.displayMode = ImageProcessor::DisplayMode::Binary;
    } else if (ui->rBtnEdges->isChecked()) {
        request.displayMode = ImageProcessor::DisplayMode::Edges;
    } else if (ui->rBtnClaheGray->isChecked()) {
        request.displayMode = ImageProcessor::DisplayMode::ClaheGray;
    } else if (ui->rBtnGrayscale->isChecked()) {
        request.displayMode = ImageProcessor::DisplayMode::Grayscale;
    } else {
        request.displayMode = ImageProcessor::DisplayMode::Original;
    }

    m_imageProcessorModule.enqueueFrame(request);
}

void Micromanipulator::on_BtnRecordVideo_clicked()
{
    if (!m_cameraModule.isCameraOpen()) {
        QMessageBox::warning(this, tr("提示"), tr("请先打开相机后再开始录屏。"));
        return;
    }

    if (m_isRecording) {
        stopRecording();
        ui->BtnRecordVideo->setText(QStringLiteral("开始录屏"));
        return;
    }

    cv::Mat referenceFrame = frame;
    if (referenceFrame.empty()) {
        if (!m_cameraModule.captureFrame(referenceFrame) || referenceFrame.empty()) {
            QMessageBox::warning(this, tr("提示"), tr("无法获取当前帧分辨率，录屏启动失败。"));
            return;
        }
    }

    const double fps = (timer1 && timer1->interval() > 0)
                           ? 1000.0 / static_cast<double>(timer1->interval())
                           : m_recordingFps;

    if (startRecording(referenceFrame.size(), fps)) {
        ui->BtnRecordVideo->setText(QStringLiteral("停止录屏"));
    } else {
        QMessageBox::warning(this, tr("提示"), tr("录屏功能启动失败，请检查目标目录是否可写。"));
    }
}

void Micromanipulator::on_BtnRecordTipError_clicked()
{
    if (m_isTipErrorRecording) {
        stopTipErrorRecording();
        ui->BtnRecordTipError->setText(QStringLiteral("开始记录误差"));
        if (!m_tipErrorFilePath.isEmpty()) {
            ui->FeedBack->setText(tr("误差记录已保存：%1").arg(m_tipErrorFilePath));
        }
        return;
    }

    if (!m_cameraModule.isCameraOpen()) {
        QMessageBox::warning(this, tr("提示"), tr("请先打开相机后再开始记录误差。"));
        return;
    }

    const QString suggestedPath = buildDefaultTipErrorLogPath();
    const QString filePath = QFileDialog::getSaveFileName(
        this,
        tr("选择误差记录文件"),
        suggestedPath,
        tr("CSV 文件 (*.csv)"));
    if (filePath.isEmpty()) {
        return;
    }

    startTipErrorRecording(filePath);
    if (m_isTipErrorRecording) {
        ui->BtnRecordTipError->setText(QStringLiteral("停止记录误差"));
        ui->FeedBack->setText(tr("误差记录中：%1").arg(m_tipErrorFilePath));
    }
}

void Micromanipulator::handleCameraOpened()
{
    // 详细注释：当 CameraModule 成功打开底层设备时，通过该槽函数更新按钮文字，
    // 并确保计时器处于工作状态，以便持续刷新界面画面。
    ui->BtnCameraOnOff->setText(QStringLiteral("关闭相机"));

}

void Micromanipulator::handleCameraClosed()
{
    // 相机关闭后停止界面更新并恢复按钮文字。
    ui->BtnCameraOnOff->setText(QStringLiteral("打开相机"));
    stopRecording();
    ui->BtnRecordVideo->setText(QStringLiteral("开始录屏"));
    stopTipErrorRecording();
    ui->BtnRecordTipError->setText(QStringLiteral("开始记录误差"));
}

void Micromanipulator::handleCameraFrame(const cv::Mat &frameFromModule)
{
    Q_UNUSED(frameFromModule);
    // 当前版本仍采用 ShowCamera 主动拉取的方式刷新界面。
    // 该槽函数仅作为接口占位，便于日后迁移到完全被动的帧分发模式。
}

void Micromanipulator::on_cBoxSmoothingMode_currentIndexChanged(int index)
{
    TipSmoothingMode newMode = TipSmoothingMode::Adaptive;
    if (index == 1) {
        newMode = TipSmoothingMode::GlobalFixed;
    } else if (index == 2) {
        newMode = TipSmoothingMode::KalmanPredictive;
    }

    if (newMode != m_tipSmoothingMode) {
        m_tipSmoothingMode = newMode;
        resetTipSmoothingState();
    }
}

// ImageProcessor::ProcessedImage Micromanipulator::runImageProcessingPipeline(const ImageProcessor::FrameRequest &request)
// {
//     ImageProcessor::ProcessedImage result;
//     cv::Mat workingFrame = request.frame.clone();
//     result.displayMode = request.displayMode;

//     if (workingFrame.empty()) {
//         return result; // 原始帧为空时直接返回默认结果，避免后续处理出错。
//     }

//     // ---------- 步骤1：根据需求执行去畸变、翻转与背景差分 ----------
//     if (request.applyCalibration && !request.cameraMatrix.empty() && !request.distCoeffs.empty()) {
//         cv::Mat map1, map2;
//         const cv::Size imageSize = workingFrame.size();
//         const cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
//             request.cameraMatrix, request.distCoeffs, imageSize, 1, imageSize, nullptr);

//         cv::initUndistortRectifyMap(request.cameraMatrix, request.distCoeffs, cv::Mat(),
//                                     newCameraMatrix, imageSize, CV_16SC2, map1, map2);
//         cv::remap(workingFrame, workingFrame, map1, map2, cv::INTER_LINEAR);
//     }

//     if (request.flipImage) {
//         cv::flip(workingFrame, workingFrame, -1); // -1 表示水平和垂直同时翻转。
//     }

//     if (!request.background.empty()) {
//         cv::Mat diff;
//         cv::absdiff(workingFrame, request.background, diff);
//         cv::bitwise_not(diff, workingFrame); // 使用反色突出变化区域。
//     }

//     // ---------- 步骤2：灰度化、去噪声、直方图均衡与二值化 ----------
//     cv::Mat gray;
//     cv::cvtColor(workingFrame, gray, cv::COLOR_BGR2GRAY);
//     cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

//     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
//     clahe->apply(gray, gray);

//     cv::Mat binaryImage;
//     // cv::threshold(gray, binaryImage, 180, 255, cv::THRESH_BINARY);
//     cv::adaptiveThreshold(gray,
//                           binaryImage,
//                           255,
//                           cv::ADAPTIVE_THRESH_GAUSSIAN_C,  // 自适应方法：高斯加权均值
//                           cv::THRESH_BINARY,     // 阈值类型：二值化（超过阈值为maxValue，否则为0）
//                           5,                     // 块大小（局部区域的尺寸，必须为奇数，如3、5、7...）
//                           1);                      // 常数C（从局部均值中减去的数值，用于调整阈值)
//     cv::medianBlur(binaryImage, binaryImage, 5);

//     // ---------- 步骤3：轮廓提取，获取关键的边缘信息 ----------
//     cv::Mat edges;
//     cv::Canny(binaryImage, edges, 150, 150);
//     cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
//     cv::erode(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

//     std::vector<std::vector<cv::Point>> contours;
//     std::vector<cv::Vec4i> hierarchy;
//     cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

//     std::vector<cv::Point> intersectionPoints; // 存储与图像边界的交点，用于在三个视图中统一标注。

//     if (!contours.empty()) {
//         size_t maxContourIndex = 0;
//         double maxContourLength = 0.0;
//         for (size_t i = 0; i < contours.size(); ++i) {
//             const double length = cv::arcLength(contours[i], true);
//             if (length > maxContourLength) {
//                 maxContourLength = length;
//                 maxContourIndex = i;
//             }
//         }

//         const std::vector<cv::Point> &longestContour = contours[maxContourIndex];

//         for (const auto &point : longestContour) {
//             if (point.x == 0 || point.x == workingFrame.cols - 1 ||
//                 point.y == 0 || point.y == workingFrame.rows - 1) {
//                 intersectionPoints.push_back(point);
//             }
//         }

//         if (!longestContour.empty()) {
//             // ---------- 针尖定位：结合主轴平滑与末端局部拟合，兼顾尖针与吸持针 ----------
//             // 1) 通过 PCA 获取整条轮廓的主轴方向，以兼容细长针与带圆角的吸持针。
//             cv::Mat contourMat(static_cast<int>(longestContour.size()), 2, CV_32F);
//             for (int row = 0; row < contourMat.rows; ++row) {
//                 contourMat.at<float>(row, 0) = static_cast<float>(longestContour[row].x);
//                 contourMat.at<float>(row, 1) = static_cast<float>(longestContour[row].y);
//             }
//             cv::PCA contourPca(contourMat, cv::Mat(), cv::PCA::DATA_AS_ROW);

//             cv::Point2f majorAxis(static_cast<float>(contourPca.eigenvectors.at<float>(0, 0)),
//                                   static_cast<float>(contourPca.eigenvectors.at<float>(0, 1)));
//             const float normMajor = std::sqrt(majorAxis.x * majorAxis.x + majorAxis.y * majorAxis.y);
//             if (normMajor > 1e-3f) {
//                 majorAxis.x /= normMajor;
//                 majorAxis.y /= normMajor;
//             } else {
//                 majorAxis = cv::Point2f(1.0f, 0.0f); // 极端退化情况下的兜底方向。
//             }

//             const cv::Point2f contourCentroid(contourPca.mean.at<float>(0, 0),
//                                               contourPca.mean.at<float>(0, 1));

//             // 2) 估计器械进入画面的基准点：优先使用与边界的交点均值，兼顾不同放置姿态。
//             cv::Point2f basePoint = contourCentroid;
//             if (!intersectionPoints.empty()) {
//                 basePoint = std::accumulate(intersectionPoints.begin(), intersectionPoints.end(),
//                                             cv::Point2f(0.0f, 0.0f),
//                                             [](const cv::Point2f &acc, const cv::Point &pt) {
//                                                 return cv::Point2f(acc.x + pt.x, acc.y + pt.y);
//                                             });
//                 basePoint.x /= static_cast<float>(intersectionPoints.size());
//                 basePoint.y /= static_cast<float>(intersectionPoints.size());
//             }

//             // 3) 调整主轴方向，使其从基准点指向器械末端方向，并结合历史方向做平滑。
//             cv::Point2f baseToCentroid = contourCentroid - basePoint;
//             if (baseToCentroid.x * majorAxis.x + baseToCentroid.y * majorAxis.y < 0.0f) {
//                 majorAxis = cv::Point2f(-majorAxis.x, -majorAxis.y);
//             }

//             if (m_hasTipAxisDirection) {
//                 // 与上一帧的主轴保持同向，避免发生 180° 翻转。
//                 if (majorAxis.x * m_tipAxisDirection.x + majorAxis.y * m_tipAxisDirection.y < 0.0f) {
//                     majorAxis = cv::Point2f(-majorAxis.x, -majorAxis.y);
//                 }

//                 // 采用指数平滑更新主轴，抑制轻微噪声对轴线的影响。
//                 cv::Point2f blended(static_cast<float>(m_tipAxisSmoothingAlpha) * majorAxis
//                                     + static_cast<float>(1.0 - m_tipAxisSmoothingAlpha) * m_tipAxisDirection);
//                 const float normBlend = std::sqrt(blended.x * blended.x + blended.y * blended.y);
//                 if (normBlend > 1e-5f) {
//                     majorAxis = cv::Point2f(blended.x / normBlend, blended.y / normBlend);
//                 }
//             } else {
//                 m_hasTipAxisDirection = true;
//             }
//             m_tipAxisDirection = majorAxis; // 记录平滑后的主轴，为下帧提供参考。

//             const cv::Point2f perpendicular(-majorAxis.y, majorAxis.x); // 与主轴垂直的单位向量。

//             // 4) 构造主轴坐标系下的投影点，便于后续根据轴向位置筛选末端区域。
//             struct AxisProjection {
//                 cv::Point2f original; ///< 原始轮廓点坐标。
//                 double axial = 0.0;   ///< 沿主轴方向的坐标。
//                 double radial = 0.0;  ///< 垂直主轴方向的坐标。
//             };

//             std::vector<AxisProjection> axisSamples;
//             axisSamples.reserve(longestContour.size());
//             double axialMax = -std::numeric_limits<double>::infinity();
//             double axialMin = std::numeric_limits<double>::infinity();

//             for (const auto &point : longestContour) {
//                 const cv::Point2f relative(static_cast<float>(point.x) - basePoint.x,
//                                            static_cast<float>(point.y) - basePoint.y);
//                 const double axial = relative.x * majorAxis.x + relative.y * majorAxis.y;
//                 if (axial < -m_tipAxialBackstopPixels) {
//                     continue; // 远离针体末端的反向点不参与末端拟合。
//                 }
//                 const double radial = relative.x * perpendicular.x + relative.y * perpendicular.y;

//                 axisSamples.push_back({cv::Point2f(static_cast<float>(point.x),
//                                                     static_cast<float>(point.y)),
//                                        axial,
//                                        radial});
//                 axialMax = std::max(axialMax, axial);
//                 axialMin = std::min(axialMin, axial);
//             }

//             if (!axisSamples.empty() && axialMax > -std::numeric_limits<double>::infinity()) {
//                 // 5) 在末端截取一段窗口，并对轴向与径向分别做抗抖动估计。
//                 const double axialRange = std::max(5.0, axialMax - std::max(0.0, axialMin));
//                 const double distalBandWidth = std::max(6.0, axialRange * 0.25); // 末端窗口宽度。

//                 std::vector<AxisProjection> distalBand;
//                 distalBand.reserve(axisSamples.size());
//                 for (const AxisProjection &sample : axisSamples) {
//                     if (sample.axial >= axialMax - distalBandWidth) {
//                         distalBand.push_back(sample);
//                     }
//                 }

//                 if (!distalBand.empty()) {
//                     const double axialCoreWidth = std::max(2.5, distalBandWidth * 0.35);
//                     std::vector<double> axialCore;
//                     std::vector<double> radialCore;
//                     axialCore.reserve(distalBand.size());
//                     radialCore.reserve(distalBand.size());
//                     std::vector<double> radialAll;
//                     radialAll.reserve(distalBand.size());

//                     for (const AxisProjection &sample : distalBand) {
//                         radialAll.push_back(sample.radial);
//                         if (sample.axial >= axialMax - axialCoreWidth) {
//                             axialCore.push_back(sample.axial);
//                             radialCore.push_back(sample.radial);
//                         }
//                     }

//                     auto trimmedMean = [](std::vector<double> values) -> double {
//                         if (values.empty()) {
//                             return 0.0;
//                         }
//                         std::sort(values.begin(), values.end());
//                         const size_t trim = values.size() / 6; // 舍弃两侧 ~16% 极端值。
//                         const size_t start = std::min(trim, values.size());
//                         const size_t end = start < values.size() ? values.size() - trim : start;
//                         if (start >= end || end > values.size()) {
//                             return values[values.size() / 2]; // 数据量太小时退回中位数。
//                         }
//                         double sum = 0.0;
//                         for (size_t idx = start; idx < end; ++idx) {
//                             sum += values[idx];
//                         }
//                         return sum / static_cast<double>(end - start);
//                     };

//                     const double axialEstimate = axialCore.empty()
//                         ? axialMax
//                         : std::accumulate(axialCore.begin(), axialCore.end(), 0.0)
//                             / static_cast<double>(axialCore.size());
//                     double radialEstimate = !radialCore.empty()
//                         ? trimmedMean(radialCore)
//                         : trimmedMean(radialAll);

//                     // 6) 依据拟合主轴构造穿过器械中心线的射线，并与轮廓求交以确定稳定末端。
//                     const cv::Point2f axisDirection = majorAxis;
//                     const cv::Point2f axisOrigin = basePoint
//                         + perpendicular * static_cast<float>(radialEstimate);

//                     auto cross2d = [](const cv::Point2f &a, const cv::Point2f &b) {
//                         return static_cast<double>(a.x) * static_cast<double>(b.y)
//                             - static_cast<double>(a.y) * static_cast<double>(b.x);
//                     };

//                     cv::Point2f estimatedTip = axisOrigin
//                         + axisDirection * static_cast<float>(axialEstimate);
//                     double bestRayT = -std::numeric_limits<double>::infinity();

//                     for (size_t idx = 0; idx < longestContour.size(); ++idx) {
//                         const cv::Point2f segmentStart(
//                             static_cast<float>(longestContour[idx].x),
//                             static_cast<float>(longestContour[idx].y));
//                         const cv::Point2f segmentEnd(
//                             static_cast<float>(longestContour[(idx + 1) % longestContour.size()].x),
//                             static_cast<float>(longestContour[(idx + 1) % longestContour.size()].y));
//                         const cv::Point2f segmentVector = segmentEnd - segmentStart;
//                         const cv::Point2f originToSegment = segmentStart - axisOrigin;

//                         const double denom = cross2d(axisDirection, segmentVector);
//                         if (std::fabs(denom) < 1e-6) {
//                             continue; // 平行或极小角度的线段不参与交点计算。
//                         }

//                         const double rayT = cross2d(originToSegment, segmentVector) / denom;
//                         const double segU = cross2d(originToSegment, axisDirection) / denom;
//                         if (rayT >= 0.0 && segU >= 0.0 && segU <= 1.0 && rayT > bestRayT) {
//                             bestRayT = rayT;
//                             estimatedTip = axisOrigin + axisDirection * static_cast<float>(rayT);
//                         }
//                     }

//                     // 7) 应用指数平滑，进一步抑制逐帧抖动并保持末端稳定。
//                     cv::Point2f filteredTip = estimatedTip;
//                     if (m_hasFilteredTipPixel) {
//                         filteredTip = static_cast<float>(m_tipSmoothingAlpha) * estimatedTip
//                             + static_cast<float>(1.0 - m_tipSmoothingAlpha) * m_filteredTipPixel;
//                     } else {
//                         m_hasFilteredTipPixel = true;
//                     }
//                     m_filteredTipPixel = filteredTip; // 保存最新的滤波结果供下帧使用。

//                     result.tipPosition = cv::Point(cvRound(filteredTip.x), cvRound(filteredTip.y));
//                     result.hasTip = true;
//                 }
//             }

//             if (!result.hasTip) {
//                 // 作为兜底策略，选择距离质心最远的点，避免完全丢失针尖。
//                 double fallbackDistance = 0.0;
//                 cv::Point fallbackPoint;
//                 for (const auto &point : longestContour) {
//                     const double distance = std::hypot(point.x - contourCentroid.x,
//                                                        point.y - contourCentroid.y);
//                     if (distance > fallbackDistance) {
//                         fallbackDistance = distance;
//                         fallbackPoint = point;
//                     }
//                 }

//                 if (fallbackDistance > 0.0) {
//                     result.tipPosition = fallbackPoint;
//                     result.hasTip = true;
//                     m_filteredTipPixel = cv::Point2f(static_cast<float>(fallbackPoint.x),
//                                                       static_cast<float>(fallbackPoint.y));
//                     m_hasFilteredTipPixel = true;
//                 }
//             }
//         }
//     }

//     // ---------- 步骤4：为三种显示模式准备可视化图像，并叠加特征点 ----------
//     auto drawFeatureMarkers = [&](cv::Mat &target) {
//         for (const auto &point : intersectionPoints) {
//             cv::circle(target, point, 5, cv::Scalar(0, 0, 255), -1); // 红色：与边界的交点。
//         }
//         if (result.hasTip) {
//             cv::circle(target, result.tipPosition, 5, cv::Scalar(255, 0, 0), -1); // 蓝色：针尖位置。
//         }
//     };

//     cv::Mat workingDisplay = workingFrame.clone();
//     drawFeatureMarkers(workingDisplay);

//     cv::Mat binaryDisplay;
//     cv::cvtColor(binaryImage, binaryDisplay, cv::COLOR_GRAY2BGR);
//     drawFeatureMarkers(binaryDisplay);

//     cv::Mat edgesDisplay;
//     cv::cvtColor(edges, edgesDisplay, cv::COLOR_GRAY2BGR);
//     drawFeatureMarkers(edgesDisplay);

//     switch (request.displayMode) {
//     case ImageProcessor::DisplayMode::Binary:
//         result.frame = binaryDisplay;
//         break;
//     case ImageProcessor::DisplayMode::Edges:
//         result.frame = edgesDisplay;
//         break;
//     case ImageProcessor::DisplayMode::WorkingFrame:
//     default:
//         result.frame = workingDisplay;
//         break;
//     }

//     return result;
// }

ImageProcessor::ProcessedImage Micromanipulator::runImageProcessingPipeline(const ImageProcessor::FrameRequest &request)
{
    // ===== 新增：跨帧针尖轨迹（静态局部，无需类成员）=====
    static std::vector<cv::Point> s_tipTrail;
    static const int s_tipTrailMaxLen = 15;

    ImageProcessor::ProcessedImage result;
    cv::Mat workingFrame = request.frame.clone();
    result.displayMode = request.displayMode;

    if (workingFrame.empty()) {
        return result; // 原始帧为空时直接返回默认结果
    }

    // ---------- 步骤1：根据需求执行去畸变、翻转与背景差分 ----------
    if (request.applyCalibration && !request.cameraMatrix.empty() && !request.distCoeffs.empty()) {
        cv::Mat map1, map2;
        if (ensureUndistortMaps(request, map1, map2)) {
            cv::remap(workingFrame, workingFrame, map1, map2, cv::INTER_LINEAR);
        }
    }

    if (request.flipImage) {
        cv::flip(workingFrame, workingFrame, -1); // -1 表示水平和垂直同时翻转
    }

    if (!request.background.empty()) {
        cv::Mat diff;
        cv::absdiff(workingFrame, request.background, diff);
        cv::bitwise_not(diff, workingFrame); // 使用反色突出变化区域
    }

    // ---------- 步骤2：灰度化、去噪声、直方图均衡与二值化 ----------
    cv::UMat gray;
    cv::cvtColor(workingFrame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    cv::UMat claheGray;
    if (!m_clahe) {
        m_clahe = cv::createCLAHE(6.0, cv::Size(8, 8));
    }
    m_clahe->apply(gray, claheGray);

    cv::UMat binaryImageGpu;
    cv::threshold(claheGray, binaryImageGpu, 180, 255, cv::THRESH_BINARY);
    cv::medianBlur(binaryImageGpu, binaryImageGpu, 5);

    cv::Mat binaryImage = binaryImageGpu.getMat(cv::ACCESS_RW);

    // 针体中存在大块高光会让内部出现大片空洞，单纯的形态学闭运算难以彻底填补。
    // 这里先将针体反转为白色，通过漫水填充将与边界相连的背景全部染白，再反转回来即可
    // 获得实心的针体二值图（内部空洞被填满）。
    {
        cv::Mat needleMask;
        cv::bitwise_not(binaryImage, needleMask);   // 先取反

        cv::Mat floodFilled = needleMask.clone();

        // 以图像右下角作为漫水填充的起始点
        cv::Point seedPt(floodFilled.cols - 1, floodFilled.rows - 1);

        cv::floodFill(floodFilled, seedPt, cv::Scalar(255));
        cv::bitwise_not(floodFilled, floodFilled);

        // 原针体 ∪ （内部空洞）= 实心针体
        needleMask |= floodFilled;
        cv::bitwise_not(needleMask, binaryImage);

    }

    // ---------- 步骤3：轮廓提取，获取关键的边缘信息 ----------
    cv::Mat edges;
    cv::Canny(binaryImage, edges, 150, 150);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    cv::erode(edges, edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Point> intersectionPoints; // 与图像边界的交点
    std::vector<cv::Point> boundaryPts;//所有边界点（存在伪边界点，需要筛选）

    // ===== 新增：本帧可视化调试量（仅用于绘制，不影响算法）=====
    std::vector<cv::Point> dbg_longestContour;
    cv::Point2f dbg_basePoint, dbg_centroid;
    cv::Point2f dbg_axisDir, dbg_axisPerp;
    bool dbg_haveAxis = false;
    std::vector<cv::Point2f> dbg_distalBandPts;
    cv::Point2f dbg_axisOrigin;
    cv::Point2f dbg_estimatedTip;
    bool dbg_haveTipGeom = false;

    if (!contours.empty()) {
        size_t maxContourIndex = 0;
        double maxContourLength = 0.0;
        for (size_t i = 0; i < contours.size(); ++i) {
            const double length = cv::arcLength(contours[i], true);
            if (length > maxContourLength) {
                maxContourLength = length;
                maxContourIndex = i;
            }
        }

        const std::vector<cv::Point> &longestContour = contours[maxContourIndex];
        dbg_longestContour = longestContour; // ===== 新增：保存最长轮廓用于绘制 =====

        for (const auto &pt : longestContour) {
            if (pt.x == 0 || pt.x == workingFrame.cols - 1 ||
                pt.y == 0 || pt.y == workingFrame.rows - 1) {
                boundaryPts.push_back(pt);
            }
        }

        if (!boundaryPts.empty()) {
            auto cmp = [](const cv::Point &a, const cv::Point &b) {
                return (a.x + a.y) < (b.x + b.y);
            };
            auto mm = std::minmax_element(boundaryPts.begin(), boundaryPts.end(), cmp);
            intersectionPoints.push_back(*mm.first);
            if (*mm.second != *mm.first) {
                intersectionPoints.push_back(*mm.second);
            }
        }

        if (!longestContour.empty()) {
            // ---------- 针尖定位：结合主轴平滑与末端局部拟合 ----------
            // 1) PCA主轴方向
            cv::Mat contourMat(static_cast<int>(longestContour.size()), 2, CV_32F);
            for (int row = 0; row < contourMat.rows; ++row) {
                contourMat.at<float>(row, 0) = static_cast<float>(longestContour[row].x);
                contourMat.at<float>(row, 1) = static_cast<float>(longestContour[row].y);
            }
            cv::PCA contourPca(contourMat, cv::Mat(), cv::PCA::DATA_AS_ROW);

            cv::Point2f majorAxis(static_cast<float>(contourPca.eigenvectors.at<float>(0, 0)),
                                  static_cast<float>(contourPca.eigenvectors.at<float>(0, 1)));
            const float normMajor = std::sqrt(majorAxis.x * majorAxis.x + majorAxis.y * majorAxis.y);
            if (normMajor > 1e-3f) {
                majorAxis.x /= normMajor;
                majorAxis.y /= normMajor;
            } else {
                majorAxis = cv::Point2f(1.0f, 0.0f); // 兜底方向
            }

            const cv::Point2f contourCentroid(contourPca.mean.at<float>(0, 0),
                                              contourPca.mean.at<float>(0, 1));
            dbg_centroid = contourCentroid; // ===== 新增：绘制质心 =====

            // 2) 基准点（优先使用与边界交点的均值）
            cv::Point2f basePoint = contourCentroid;
            if (!intersectionPoints.empty()) {
                basePoint = std::accumulate(intersectionPoints.begin(), intersectionPoints.end(),
                                            cv::Point2f(0.0f, 0.0f),
                                            [](const cv::Point2f &acc, const cv::Point &pt) {
                                                return cv::Point2f(acc.x + pt.x, acc.y + pt.y);
                                            });
                basePoint.x /= static_cast<float>(intersectionPoints.size());
                basePoint.y /= static_cast<float>(intersectionPoints.size());
            }
            dbg_basePoint = basePoint; // ===== 新增：绘制基准点 =====

            // 额外的方向纠正：确保轴向正方向指向距离基准点更远的一侧
            //（若负向投影的最大幅值大于正向，则翻转，以降低基准点与质心重合时的随机性）
            double axialMinRaw =  std::numeric_limits<double>::infinity();
            double axialMaxRaw = -std::numeric_limits<double>::infinity();
            cv::Point2f farthestVec(0.f, 0.f);
            for (const auto &p : longestContour) {
                const cv::Point2f rel = cv::Point2f((float)p.x, (float)p.y) - basePoint;
                const double axial = rel.x * majorAxis.x + rel.y * majorAxis.y;
                axialMinRaw = std::min(axialMinRaw, axial);
                axialMaxRaw = std::max(axialMaxRaw, axial);

                // 记录投影绝对值最大的点，作为方向证据
                if (std::fabs(axial) > std::fabs(farthestVec.x * majorAxis.x + farthestVec.y * majorAxis.y)) {
                    farthestVec = rel;
                }
            }
            if (axialMaxRaw < std::fabs(axialMinRaw)) {
                majorAxis = cv::Point2f(-majorAxis.x, -majorAxis.y);
            }
            // 以“离基准点最远的投影”作为指向提示，保证翻转时能随视角变化及时调整
            const float farthestNorm = std::sqrt(farthestVec.x * farthestVec.x + farthestVec.y * farthestVec.y);
            if (farthestNorm > 1e-3f) {
                cv::Point2f axisHint = cv::Point2f(farthestVec.x / farthestNorm, farthestVec.y / farthestNorm);
                if (majorAxis.x * axisHint.x + majorAxis.y * axisHint.y < 0.0f) {
                    majorAxis = cv::Point2f(-majorAxis.x, -majorAxis.y);
                }
            }

            // // ===== 新增：导出到桌面 CSV（Excel 可打开）=====
            // // 说明：每帧追加一行，包含：时间戳、帧号、图像尺寸、四个边界角点、基准点、质心、主轴方向、与边界交点统计
            // static bool s_logInited = false;
            // static QFile s_logFile;
            // static QTextStream s_logOut;
            // static qint64 s_frameIndex = 0;

            // // 1) 首次初始化：创建桌面文件并写表头
            // if (!s_logInited) {
            //     const QString desktopPath = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation);
            //     const QString filePath = QDir(desktopPath).filePath(
            //         QString("VisionBoundaryBaseLog_%1.csv")
            //             .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")));
            //     s_logFile.setFileName(filePath);
            //     if (s_logFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            //         s_logOut.setDevice(&s_logFile);
            //         s_logOut.setRealNumberPrecision(6); // 小数位精度
            //         // 表头
            //         s_logOut
            //             << "timestamp" << ','
            //             << "frame_index" << ','
            //             << "img_w" << ',' << "img_h" << ','
            //             << "corner_TL_x" << ',' << "corner_TL_y" << ','
            //             << "corner_TR_x" << ',' << "corner_TR_y" << ','
            //             << "corner_BR_x" << ',' << "corner_BR_y" << ','
            //             << "corner_BL_x" << ',' << "corner_BL_y" << ','
            //             << "base_x" << ',' << "base_y" << ',' << "base_side" << ','
            //             << "centroid_x" << ',' << "centroid_y" << ','
            //             << "axis_dir_x" << ',' << "axis_dir_y" << ','
            //             << "intersections_count" << ','
            //             << "intersections_xy(list)" << ','
            //             << "intersections_mean_x" << ','
            //             << "intersections_mean_y"
            //             << '\n';
            //         s_logOut.flush();
            //         s_logInited = true;
            //     }
            // }

            // // 2) 计算边界角点（图像固定四角），判定基准点落在哪一侧（Left/Right/Top/Bottom/Inside）
            // const int W = workingFrame.cols;
            // const int H = workingFrame.rows;
            // const cv::Point2f cornerTL(0.f, 0.f);
            // const cv::Point2f cornerTR((float)(W - 1), 0.f);
            // const cv::Point2f cornerBR((float)(W - 1), (float)(H - 1));
            // const cv::Point2f cornerBL(0.f, (float)(H - 1));

            // // 判定基准点所在边（带1px裕量）
            // QString baseSide = "Inside";
            // if (std::fabs(dbg_basePoint.x - 0.f) <= 1.f)            baseSide = "Left";
            // else if (std::fabs(dbg_basePoint.x - (W - 1)) <= 1.f)   baseSide = "Right";
            // else if (std::fabs(dbg_basePoint.y - 0.f) <= 1.f)       baseSide = "Top";
            // else if (std::fabs(dbg_basePoint.y - (H - 1)) <= 1.f)   baseSide = "Bottom";

            // // 3) 统计与边界交点的均值，便于和基准点对比
            // double interMeanX = 0.0, interMeanY = 0.0;
            // if (!intersectionPoints.empty()) {
            //     for (const auto& ip : intersectionPoints) {
            //         interMeanX += ip.x;
            //         interMeanY += ip.y;
            //     }
            //     interMeanX /= (double)intersectionPoints.size();
            //     interMeanY /= (double)intersectionPoints.size();
            // }

            // // 4) 序列化交点列表为 "x1 y1 | x2 y2 | ..." 方便 Excel 观测
            // QString interList;
            // if (!intersectionPoints.empty()) {
            //     interList.reserve(intersectionPoints.size() * 12);
            //     for (size_t i = 0; i < intersectionPoints.size(); ++i) {
            //         interList += QString::number(intersectionPoints[i].x);
            //         interList += ' ';
            //         interList += QString::number(intersectionPoints[i].y);
            //         if (i + 1 < intersectionPoints.size()) interList += " | ";
            //     }
            // }

            // // 5) 写一行 CSV
            // if (s_logInited) {
            //     s_logOut
            //         << QDateTime::currentDateTime().toString(Qt::ISODateWithMs) << ','
            //         << s_frameIndex++ << ','
            //         << W << ',' << H << ','
            //         << cornerTL.x << ',' << cornerTL.y << ','
            //         << cornerTR.x << ',' << cornerTR.y << ','
            //         << cornerBR.x << ',' << cornerBR.y << ','
            //         << cornerBL.x << ',' << cornerBL.y << ','
            //         << dbg_basePoint.x << ',' << dbg_basePoint.y << ','
            //         << baseSide << ','
            //         << dbg_centroid.x << ',' << dbg_centroid.y << ','
            //         << dbg_axisDir.x << ',' << dbg_axisDir.y << ','
            //         << (int)intersectionPoints.size() << ','
            //         << '"' << interList << '"' << ','
            //         << interMeanX << ','
            //         << interMeanY
            //         << '\n';
            //     // 为了稳妥，逐帧 flush（如果担心性能可改为每N帧）
            //     s_logOut.flush();
            // }
            // // ===== 新增导出结束 =====


            cv::Point2f baseToCentroid = contourCentroid - basePoint;
            if (baseToCentroid.x * majorAxis.x + baseToCentroid.y * majorAxis.y < 0.0f) {
                majorAxis = cv::Point2f(-majorAxis.x, -majorAxis.y);
            }
            if (m_hasTipAxisDirection) {
                const float historyDot = majorAxis.x * m_tipAxisDirection.x + majorAxis.y * m_tipAxisDirection.y;

                // 若新观测与历史完全相反（如旋转 180°），直接采信观测，允许快速翻转
                if (historyDot < -0.2f) {
                    majorAxis = majorAxis; // 保持数据驱动方向，不与历史对齐
                } else {
                    cv::Point2f aligned = (historyDot < 0.0f)
                                              ? cv::Point2f(-majorAxis.x, -majorAxis.y)
                                              : majorAxis;
                    cv::Point2f blended(static_cast<float>(m_tipAxisSmoothingAlpha) * aligned
                                        + static_cast<float>(1.0 - m_tipAxisSmoothingAlpha) * m_tipAxisDirection);
                    const float normBlend = std::sqrt(blended.x * blended.x + blended.y * blended.y);
                    if (normBlend > 1e-5f) {
                        majorAxis = cv::Point2f(blended.x / normBlend, blended.y / normBlend);
                    }
                }
            } else {
                m_hasTipAxisDirection = true;
            }
            m_tipAxisDirection = majorAxis;

            const cv::Point2f perpendicular(-majorAxis.y, majorAxis.x);
            dbg_axisDir  = majorAxis;     // ===== 新增：绘制主轴 =====
            dbg_axisPerp = perpendicular; // ===== 新增：绘制法向 =====
            dbg_haveAxis = true;

            // 4) 轴向/径向投影，并构造“末端带”
            struct AxisProjection { cv::Point2f original; double axial=0.0; double radial=0.0; };
            std::vector<AxisProjection> axisSamples;
            axisSamples.reserve(longestContour.size());
            double axialMax = -std::numeric_limits<double>::infinity();
            double axialMin = std::numeric_limits<double>::infinity();

            for (const auto &point : longestContour) {
                const cv::Point2f relative(static_cast<float>(point.x) - basePoint.x,
                                           static_cast<float>(point.y) - basePoint.y);
                const double axial = relative.x * majorAxis.x + relative.y * majorAxis.y;
                if (axial < -m_tipAxialBackstopPixels) {
                    continue; // 远离末端的反向点不参与
                }
                const double radial = relative.x * perpendicular.x + relative.y * perpendicular.y;
                axisSamples.push_back({cv::Point2f((float)point.x, (float)point.y), axial, radial});
                axialMax = std::max(axialMax, axial);
                axialMin = std::min(axialMin, axial);
            }

            cv::Point2f measuredTip;
            bool haveMeasuredTip = false;

            if (!axisSamples.empty() && axialMax > -std::numeric_limits<double>::infinity()) {
                const double axialRange = std::max(5.0, axialMax - std::max(0.0, axialMin));
                const double distalBandWidth = std::max(6.0, axialRange * 0.1);

                std::vector<AxisProjection> distalBand;
                distalBand.reserve(axisSamples.size());
                for (const AxisProjection &s : axisSamples) {
                    if (s.axial >= axialMax - distalBandWidth) {
                        distalBand.push_back(s);
                        dbg_distalBandPts.push_back(s.original); // ===== 新增：末端带样本（图像域）=====
                    }
                }

                if (!distalBand.empty()) {
                    const double axialCoreWidth = std::max(2.5, distalBandWidth * 0.35);
                    std::vector<double> axialCore, radialCore, radialAll;
                    axialCore.reserve(distalBand.size());
                    radialCore.reserve(distalBand.size());
                    radialAll.reserve(distalBand.size());

                    for (const AxisProjection &s : distalBand) {
                        radialAll.push_back(s.radial);
                        if (s.axial >= axialMax - axialCoreWidth) {
                            axialCore.push_back(s.axial);
                            radialCore.push_back(s.radial);
                        }
                    }

                    auto trimmedMean = [](std::vector<double> v)->double {
                        if (v.empty()) return 0.0;
                        std::sort(v.begin(), v.end());
                        const size_t trim = v.size()/6;
                        const size_t start = std::min(trim, v.size());
                        const size_t end = start < v.size() ? v.size()-trim : start;
                        if (start >= end || end > v.size()) return v[v.size()/2];
                        double s = 0.0; for (size_t i=start;i<end;++i) s += v[i];
                        return s / double(end-start);
                    };

                    const double axialEstimate = axialCore.empty()
                                                     ? axialMax
                                                     : std::accumulate(axialCore.begin(), axialCore.end(), 0.0) / double(axialCore.size());

                    double radialEstimate = !radialCore.empty() ? trimmedMean(radialCore)
                                                                : trimmedMean(radialAll);

                    // 5) 几何求交：中心线射线与轮廓
                    const cv::Point2f axisDirection = majorAxis;
                    const cv::Point2f axisOrigin = basePoint + perpendicular * (float)radialEstimate;
                    dbg_axisOrigin = axisOrigin; // ===== 新增：射线起点 =====

                    auto cross2d = [](const cv::Point2f &a, const cv::Point2f &b){
                        return (double)a.x*b.y - (double)a.y*b.x;
                    };

                    cv::Point2f estimatedTip = axisOrigin + axisDirection * (float)axialEstimate;
                    double bestRayT = -std::numeric_limits<double>::infinity();

                    for (size_t i = 0; i < longestContour.size(); ++i) {
                        const cv::Point2f p0((float)longestContour[i].x,
                                             (float)longestContour[i].y);
                        const cv::Point2f p1((float)longestContour[(i+1)%longestContour.size()].x,
                                             (float)longestContour[(i+1)%longestContour.size()].y);
                        const cv::Point2f seg  = p1 - p0;
                        const cv::Point2f w    = p0 - axisOrigin;

                        const double denom = cross2d(axisDirection, seg);
                        if (std::fabs(denom) < 1e-6) continue;

                        const double rayT = cross2d(w, seg) / denom;
                        const double segU = cross2d(w, axisDirection) / denom;
                        if (rayT >= 0.0 && segU >= 0.0 && segU <= 1.0 && rayT > bestRayT) {
                            bestRayT = rayT;
                            estimatedTip = axisOrigin + axisDirection * (float)rayT;
                        }
                    }

                    measuredTip = estimatedTip;
                    haveMeasuredTip = true;

                    // ===== 新增：保存几何求交末端（未平滑）用于展示 =====
                    dbg_estimatedTip = estimatedTip;
                    dbg_haveTipGeom = true;
                }
            }

            if (!haveMeasuredTip) {
                // 兜底：距离质心最远点
                double fallbackDistance = 0.0;
                cv::Point fallbackPoint;
                for (const auto &p : longestContour) {
                    const double d = std::hypot(p.x - dbg_centroid.x, p.y - dbg_centroid.y);
                    if (d > fallbackDistance) { fallbackDistance = d; fallbackPoint = p; }
                }

                if (fallbackDistance > 0.0) {
                    measuredTip = cv::Point2f((float)fallbackPoint.x, (float)fallbackPoint.y);
                    haveMeasuredTip = true;
                }
            }

            if (haveMeasuredTip) {
                const cv::Point2f filteredTip = smoothTip(measuredTip, true);
                result.filteredTip = filteredTip;
                result.measuredTip = measuredTip;
                result.hasMeasuredTip = true;
                result.tipPosition = cv::Point(cvRound(filteredTip.x), cvRound(filteredTip.y));
                result.hasTip = true;

                s_tipTrail.push_back(result.tipPosition);
                if ((int)s_tipTrail.size() > s_tipTrailMaxLen) {
                    s_tipTrail.erase(s_tipTrail.begin());
                }
            }
        }
    }

    // ---------- 步骤4：为三种显示模式准备可视化图像，并叠加特征点 ----------
    auto drawFeatureMarkers = [&](cv::Mat &target) {

        // 与图像边界的交点（红色）
        if(ui->cBoxIntersection->isChecked()){
            for (const auto &pt : intersectionPoints) {
                cv::circle(target, pt, 4, cv::Scalar(0, 0, 255), -1);
            }
        }

        // 最长轮廓（绿色）
        if(ui->cBoxLongest->isChecked()){
            if (!dbg_longestContour.empty()) {
                std::vector<std::vector<cv::Point>> polys(1, dbg_longestContour);
                cv::polylines(target, polys, true, cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
            }
        }

        // 基准点（洋红）、质心（青色）、PCA主轴与法向
        if(ui->cBoxMainPoint->isChecked()){
            if (dbg_haveAxis) {
                cv::drawMarker(target, dbg_basePoint, cv::Scalar(255, 0, 255),
                               cv::MARKER_CROSS, 14, 2);
                cv::drawMarker(target, dbg_centroid, cv::Scalar(255, 255, 0),
                               cv::MARKER_TILTED_CROSS, 14, 2);

                // 主轴（黄箭头）& 法向（浅黄线），过质心绘制
                const float L = 0.25f * std::min(target.cols, target.rows);
                cv::Point2f a1 = dbg_centroid - dbg_axisDir  * L;
                cv::Point2f a2 = dbg_centroid + dbg_axisDir  * L;
                cv::Point2f b1 = dbg_centroid - dbg_axisPerp * (L * 0.6f);
                cv::Point2f b2 = dbg_centroid + dbg_axisPerp * (L * 0.6f);
                cv::arrowedLine(target, a1, a2, cv::Scalar(0, 255, 255), 2, cv::LINE_AA, 0, 0.2);
                cv::line(target, b1, b2, cv::Scalar(200, 200, 0), 1, cv::LINE_AA);
            }
        }

        // 末端带样本（黄色小点）
        if(ui->cBoxDistalBand->isChecked()){
            for (const auto &p : dbg_distalBandPts) {
                cv::circle(target, p, 2, cv::Scalar(0, 255, 255), -1);
            }
        }

        // 中心线射线（品红）与几何求交末端（橙色）
        if(ui->cBoxCenterLine->isChecked()){
            if (dbg_haveAxis && dbg_haveTipGeom) {
                const float bigL = 2.0f * std::max(target.cols, target.rows);
                cv::Point2f rayEnd = dbg_axisOrigin + dbg_axisDir * bigL;
                cv::line(target, dbg_axisOrigin, rayEnd, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
                cv::circle(target, dbg_estimatedTip, 4, cv::Scalar(0, 165, 255), -1);
            }
        }

        // 平滑后的针尖（蓝点）与跨帧轨迹（蓝线）
        if(ui->cBoxTipLine->isChecked()){
            if (result.hasTip) {
                cv::circle(target, result.tipPosition, 5, cv::Scalar(255, 0, 0), -1);
            }
            if (!s_tipTrail.empty()) {
                for (size_t i = 1; i < s_tipTrail.size(); ++i) {
                    cv::line(target, s_tipTrail[i - 1], s_tipTrail[i],
                             cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                }
            }}
    };

    cv::Mat originalDisplay = workingFrame.clone();
    cv::Mat grayDisplay;
    cv::cvtColor(gray, grayDisplay, cv::COLOR_GRAY2BGR);
    cv::Mat claheDisplay;
    cv::cvtColor(claheGray, claheDisplay, cv::COLOR_GRAY2BGR);
    cv::Mat binaryDisplay;
    cv::cvtColor(binaryImage, binaryDisplay, cv::COLOR_GRAY2BGR);
    cv::Mat edgesDisplay;
    cv::cvtColor(edges, edgesDisplay, cv::COLOR_GRAY2BGR);

    drawFeatureMarkers(originalDisplay);
    drawFeatureMarkers(grayDisplay);
    drawFeatureMarkers(claheDisplay);
    drawFeatureMarkers(binaryDisplay);
    drawFeatureMarkers(edgesDisplay);

    switch (request.displayMode) {
    case ImageProcessor::DisplayMode::Binary:
        result.frame = binaryDisplay;
        break;
    case ImageProcessor::DisplayMode::Edges:
        result.frame = edgesDisplay;
        break;
    case ImageProcessor::DisplayMode::ClaheGray:
        result.frame = claheDisplay;
        break;
    case ImageProcessor::DisplayMode::Grayscale:
        result.frame = grayDisplay;
        break;
    case ImageProcessor::DisplayMode::Original:
    default:
        result.frame = originalDisplay;
        break;
    }

    return result;
}

bool Micromanipulator::ensureUndistortMaps(const ImageProcessor::FrameRequest &request,
                                           cv::Mat &map1, cv::Mat &map2)
{
    const cv::Size imageSize = request.frame.size();
    const bool sizeChanged = imageSize != m_cachedUndistortSize;
    const bool cameraChanged = m_cachedCameraMatrix.empty() || m_cachedDistCoeffs.empty()
        || cv::norm(m_cachedCameraMatrix, request.cameraMatrix, cv::NORM_INF) > 1e-9
        || cv::norm(m_cachedDistCoeffs, request.distCoeffs, cv::NORM_INF) > 1e-9;

    if (sizeChanged || cameraChanged || m_cachedUndistortMap1.empty() || m_cachedUndistortMap2.empty()) {
        const cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
            request.cameraMatrix, request.distCoeffs, imageSize, 1, imageSize, nullptr);

        cv::initUndistortRectifyMap(request.cameraMatrix, request.distCoeffs, cv::Mat(),
                                    newCameraMatrix, imageSize, CV_16SC2,
                                    m_cachedUndistortMap1, m_cachedUndistortMap2);

        m_cachedCameraMatrix = request.cameraMatrix.clone();
        m_cachedDistCoeffs = request.distCoeffs.clone();
        m_cachedUndistortSize = imageSize;
    }

    map1 = m_cachedUndistortMap1;
    map2 = m_cachedUndistortMap2;
    return !map1.empty() && !map2.empty();
}


void Micromanipulator::handleProcessedImage(const ImageProcessor::ProcessedImage &result)
{
    if (result.frame.empty()) {
        return;
    }

    cv::Mat displayFrame = result.frame; // 共享引用，避免在 UI 线程重复拷贝大帧数据。

    if (ui->checkFocusRoiOverlay->isChecked() && m_hasLastFocusRoi && !displayFrame.empty()) {
        const cv::Scalar roiColor = m_lastFocusRoiAnchoredToTip ? cv::Scalar(0, 255, 255)
                                                                : cv::Scalar(0, 165, 255);
        cv::rectangle(displayFrame, m_lastFocusRoi, roiColor, 2, cv::LINE_AA);

        const cv::Point smoothedCenter(static_cast<int>(std::round(m_filteredFocusCenter.x)),
                                       static_cast<int>(std::round(m_filteredFocusCenter.y)));
        cv::drawMarker(displayFrame, smoothedCenter, cv::Scalar(0, 180, 255), cv::MARKER_CROSS, 14, 2, cv::LINE_AA);

        const cv::Point rawCenter(static_cast<int>(std::round(m_lastRawFocusCenter.x)),
                                  static_cast<int>(std::round(m_lastRawFocusCenter.y)));
        cv::circle(displayFrame, rawCenter, 4, cv::Scalar(200, 200, 255), 2, cv::LINE_AA);

        const QString anchorText = m_lastFocusRoiAnchoredToTip
                                       ? tr("ROI 基于针尖追踪")
                                       : tr("ROI 回退-保留上一位置");
        cv::putText(displayFrame,
                    anchorText.toStdString(),
                    cv::Point(std::max(0, m_lastFocusRoi.x), std::max(20, m_lastFocusRoi.y - 8)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    roiColor,
                    1,
                    cv::LINE_AA);
    }

    m_lastDisplayedFrame = displayFrame; // 让 m_lastDisplayedFrame 持有同一份图像数据，延长生命周期以供绘制。

    const QImage qimg(displayFrame.data, displayFrame.cols, displayFrame.rows,
                      displayFrame.step, QImage::Format_BGR888);
    ui->CameraShow->setPixmap(QPixmap::fromImage(qimg));

    if (m_isRecording && m_videoWriter.isOpened()) {
        if (displayFrame.size() == m_recordingFrameSize) {
            m_videoWriter.write(displayFrame);
        } else {
            qWarning() << "录屏被跳过：当前帧分辨率" << displayFrame.size().width << "x" << displayFrame.size().height
                       << "与录屏设定的分辨率" << m_recordingFrameSize.width << "x" << m_recordingFrameSize.height << "不一致";
        }
    }

    if (m_isTipErrorRecording) {
        writeTipErrorSample(result);
    }

    if (!result.hasTip) {
        resetTipSmoothingState();
        return;
    }

    VisualPosition_X = result.tipPosition.x;
    VisualPosition_Y = result.tipPosition.y;
    ui->cameraTipPosition->setText("(" + QString::number(VisualPosition_X) + "," +
                                   QString::number(VisualPosition_Y) + ")");

    if (TransMatGet == true) {
        double X = a * VisualPosition_X + b * VisualPosition_Y + c;
        double Y = d * VisualPosition_X + e * VisualPosition_Y + f;
        RobotPosition_X = static_cast<int>(X);
        RobotPosition_Y = static_cast<int>(Y);

        ui->robotTipPosition->setText("(" + QString::number(RobotPosition_X) + "," +
                                      QString::number(RobotPosition_Y) + ")");
    }

    if (!H1.empty()) {
        cv::Mat Visual = (cv::Mat_<double>(3, 1) << VisualPosition_X, VisualPosition_Y, 1);
        cv::Mat Arm = H1 * Visual;
        cv::Point2f armCoord(Arm.at<double>(0, 0) / Arm.at<double>(2, 0),
                             Arm.at<double>(1, 0) / Arm.at<double>(2, 0));
        ui->robotTipPosition->setText("映射机械臂：(" + QString::number(std::round(armCoord.x)) +
                                      "," + QString::number(std::round(armCoord.y)) + ")");
    }
}

bool Micromanipulator::startRecording(const cv::Size &frameSize, double fps)
{
    if (frameSize.width <= 0 || frameSize.height <= 0) {
        qWarning() << "startRecording -> 无效的帧尺寸" << frameSize.width << frameSize.height;
        return false;
    }

    stopRecording();

    QDir targetDir("C:/Users/Lee/Desktop");
    if (!targetDir.exists()) {
        targetDir.mkpath(".");
    }

    const QString filename = QStringLiteral("camera_record_%1.avi")
                                 .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    const QString filePath = targetDir.filePath(filename);

    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    if (!m_videoWriter.open(filePath.toStdString(), fourcc, fps, frameSize)) {
        qWarning() << "startRecording -> 打开录屏文件失败" << filePath;
        return false;
    }

    m_isRecording = true;
    m_recordingFrameSize = frameSize;
    m_recordingFps = fps;
    m_recordingFilePath = filePath;
    qDebug() << "录屏已开始，输出路径:" << filePath;
    return true;
}

void Micromanipulator::stopRecording()
{
    if (m_isRecording) {
        m_videoWriter.release();
        qDebug() << "录屏已停止，文件保存至" << m_recordingFilePath;
    }
    m_isRecording = false;
    m_recordingFrameSize = cv::Size();
    m_recordingFilePath.clear();
}

void Micromanipulator::resetTipSmoothingState()
{
    m_hasFilteredTipPixel = false;
    m_hasTipAxisDirection = false;
    m_tipAxisDirection = cv::Point2f(1.0f, 0.0f);
    m_hasLastMeasuredTip = false;
    m_kalmanInitialized = false;
    m_kalmanProcessNoisePos = 2e-3f;
    m_kalmanProcessNoiseVel = 5e-3f;
    m_kalmanMeasurementNoise = 8e-3f;
    m_kalmanInitialError = 1.0f;
    m_tipKalmanFilter = cv::KalmanFilter();
}

//提供设备支持的分辨率与帧率填入列表中
void Micromanipulator::updateResolutionAndFrameRates()
{
    ui->CboxResolutionAndFrameRates->clear();

    const auto cameras = QMediaDevices::videoInputs();
    int cameraIndex = ui->CboxCameraDevices->currentIndex();
    if (cameraIndex < 0 || cameraIndex >= cameras.size()) return;

    const QCameraDevice &cameraDevice = cameras.at(cameraIndex);
    QList<QCameraFormat> formats = cameraDevice.videoFormats();
    // QStringList existingOptions;  // 用于存储已添加的选项文本

    for (const QCameraFormat &format : formats) {
        QSize resolution = format.resolution();
        float minFrameRate = format.minFrameRate();
        float maxFrameRate = format.maxFrameRate();

        QString optionText = QString("%1 x %2, %3 - %4 FPS")
                                 .arg(resolution.width())
                                 .arg(resolution.height())
                                 .arg(minFrameRate)
                                 .arg(maxFrameRate);
        ui->CboxResolutionAndFrameRates->addItem(optionText);
        // 检查是否已存在该选项
        // if (!existingOptions.contains(optionText)) {
        //     ui->CboxResolutionAndFrameRates->addItem(optionText);
        //     existingOptions.append(optionText);  // 添加到新列表中以跟踪已添加的选项
        // }
    }
}

//更新设备分辨率和帧率
void Micromanipulator::on_BtnSetResolutionAndFrameRate_clicked()
{
    //更新相机分辨率与帧率
    int index = ui->CboxResolutionAndFrameRates->currentIndex();
    const auto cameras = QMediaDevices::videoInputs();
    int cameraIndex = ui->CboxCameraDevices->currentIndex();

    if (index >= 0 && cameraIndex >= 0 && cameraIndex < cameras.size()) {
        const QCameraDevice &cameraDevice = cameras.at(cameraIndex);
        QCameraFormat selectedFormat = cameraDevice.videoFormats().at(index);

        m_cameraModule.setProperty(cv::CAP_PROP_FRAME_WIDTH, selectedFormat.resolution().width());
        m_cameraModule.setProperty(cv::CAP_PROP_FRAME_HEIGHT, selectedFormat.resolution().height());
        m_cameraModule.setProperty(cv::CAP_PROP_FPS, selectedFormat.maxFrameRate());

        qDebug() << "设置分辨率和帧率：" << selectedFormat.resolution()
                 << "帧率：" << selectedFormat.maxFrameRate();
        QSize size = selectedFormat.resolution();
        ui->FeedBack->setText(QString("分辨率：%1 x %2").arg(size.width()).arg(size.height()));
    }
}

//串口参数初始化
bool Micromanipulator::getSerialPortConfig()
{
    //===================== 微动机械臂串口参数整合入口 =====================//
    // 统一读取 UI 文本后交由 MicroArm::applySerialSettings 解析，避免界面层堆砌枚举转换逻辑。
    mPortName = ui->CboxSerialPort->currentText();
    mBaudRate = ui->CboxBaudRate->currentText();
    mParity = ui->CboxParity->currentText();
    mDataBits = ui->CboxDataBits->currentText();
    mStopBits = ui->CboxStopBits->currentText();

    QString errorMessage;
    if (!m_microArmModule.applySerialSettings(mPortName,
                                              mBaudRate,
                                              mParity,
                                              mDataBits,
                                              mStopBits,
                                              &errorMessage))
    {
        ui->FeedBack->setText(errorMessage);
        return false;
    }

    return true;
}

//串口开关触发
void Micromanipulator::on_BtnSerialPortOnOff_clicked()
{
    auto setMicroJogUiEnabled = [this](bool enabled) {
        ui->BtnMicroJogUp->setEnabled(enabled);
        ui->BtnMicroJogDown->setEnabled(enabled);
        ui->BtnMicroJogLeft->setEnabled(enabled);
        ui->BtnMicroJogRight->setEnabled(enabled);
        ui->spinBoxMicroJogStep->setEnabled(enabled);
        if (!enabled) {
            stopMicroJog();
        }
    };

    if(mIsOpen == true)
    {
        m_microArmModule.closePort();
        ui->BtnSerialPortOnOff->setText("打开");
        mIsOpen = false;
        ui->BtnSerialPortSend->setEnabled(false);
        ui->BtnGetCurrentPosition->setEnabled(false);
        ui->BtnMove->setEnabled(false);
        ui->BtnSetVR->setEnabled(false);
        ui->CboxSerialPort->setEnabled(true);
        ui->CboxBaudRate->setEnabled(true);
        ui->CboxParity->setEnabled(true);
        ui->CboxDataBits->setEnabled(true);
        ui->CboxStopBits->setEnabled(true);
        ui->rBtnAbsoluteMove->setEnabled(false);
        ui->rBtnRelativeMove->setEnabled(false);
        ui->BtnInterruptMove->setEnabled(false);
        ui->BtnSetOrigin->setEnabled(false);
        ui->BtnGetTrans->setEnabled(false);
        ui->FeedBack->clear();
        ui->FeedBack->append("串口已关闭");
        setMicroJogUiEnabled(false);
    }
    else
    {
        if(getSerialPortConfig() == true)
        {
            QString openError;
            if (!m_microArmModule.openConfiguredPort(&openError))
            {
                ui->FeedBack->setText(openError);
                mIsOpen = false;
                return;
            }

            mIsOpen = true;
            ui->BtnSerialPortOnOff->setText("断开");
            qDebug() << "成功打开串口"<<mPortName;
            ui->FeedBack->append("成功打开串口"+mPortName);
            // 将串口参数选项禁用
            ui->BtnSerialPortSend->setEnabled(true);
            ui->BtnGetCurrentPosition->setEnabled(true);
            ui->BtnMove->setEnabled(true);
            ui->BtnSetVR->setEnabled(true);
            ui->CboxSerialPort->setEnabled(false);
            ui->CboxBaudRate->setEnabled(false);
            ui->CboxParity->setEnabled(false);
            ui->CboxDataBits->setEnabled(false);
            ui->CboxStopBits->setEnabled(false);
            ui->rBtnAbsoluteMove->setEnabled(true);
            ui->rBtnAbsoluteMove->click();
            ui->rBtnRelativeMove->setEnabled(true);
            ui->BtnInterruptMove->setEnabled(true);
            ui->BtnSetOrigin->setEnabled(true);
            ui->BtnGetTrans->setEnabled(true);
            setMicroJogUiEnabled(true);
        }
        else
        {
            mIsOpen = false;
        }
    }
}

//获取当前位置按钮
//========================= 获取微动机械臂当前位置 =========================//
// 功能：发送 0x63 指令请求微动机械臂反馈坐标，并启动“异步等待”逻辑。
//      串口数据由 SerialReadData 分批接收，因此这里只负责清空缓存并触发发送。
// 备注：真正的解析流程见 handleMicPositionFrame()，以保证即便数据被拆成多段
//       也能在串口回调中正确拼接和解析。
void Micromanipulator::on_BtnGetCurrentPosition_clicked()
{
    if (mIsOpen)
    {
        // 1) 清空旧的缓冲，保证不会混入上一次的残留数据片段。
        MicInfo.clear();
        m_pendingMicPositionBytes.clear();
        m_waitingForMicPositionResponse = true;

        // 2) 提示 UI 正在等待坐标返回，并下发原协议的 0x63 0x0D 指令。
        ui->FeedBack->setText("Getting CurrentPosition");
        if (!m_microArmModule.requestCurrentPosition())
        {
            ui->FeedBack->setText("获取当前位置指令发送失败，请检查串口连接");
            m_waitingForMicPositionResponse = false;
        }
    }
}

//移动至指定位置
void Micromanipulator::on_BtnMove_clicked()
{
    if(mIsOpen == true)
    {
        bool ok1, ok2, ok3;
        QString  data_1 = ui->SetXPosition->text();
        QString  data_2 = ui->SetYPosition->text();
        QString  data_3 = ui->SetZPosition->text();

        qint32 data_x = data_1.toInt(&ok1);
        qint32 data_y = data_2.toInt(&ok2);
        qint32 data_z = data_3.toInt(&ok3);

        //如果xyz输入不是数字，数据类型转换失败
        if(!ok1 || !ok2 || !ok3)
        {
            ui->FeedBack->setText("error:xyz坐标数据转换失败，请检查数据类型！");
        }
        //判定是否超出行程
        else if(data_x > 300000 || data_y > 300000 || data_z > 300000
                 || data_x < -300000 || data_y < -300000 || data_z < -300000 )//这里设置微步为0.04um,250000对应的位置为 0.04*250000 = 10000um = 10mm
        {

            ui->FeedBack->setText("error:坐标超出移动范围！");

        }
        else
        {
            if (!m_microArmModule.moveToPose(data_x, data_y, data_z))
            {
                ui->FeedBack->setText("微动机械臂位置指令发送失败");
                return;
            }

            mic_X = data_x;
            mic_Y = data_y;
            mic_Z = data_z;
        }
    }

}

// 辅助函数：实现16进制字符串按字节反转,用于移动指令
QString Micromanipulator::swapBytes_1(qint32 data)
{
    // 将 qint32 强制转换为 quint32，自动获取补码表示（这里主要是为了便于处理负位移的指令）
    quint32 unsignedData = static_cast<quint32>(data);
    // 确保字符串长度是偶数，并且填充为 8 个字符
    QString hexStr = QString::number(unsignedData, 16).toUpper().rightJustified(8, '0');


    // 按照两个字符为一组进行拆分
    QString byte1 = hexStr.mid(0, 2);
    QString byte2 = hexStr.mid(2, 2);
    QString byte3 = hexStr.mid(4, 2);
    QString byte4 = hexStr.mid(6, 2);

    // 按照倒序拼接
    return byte4 + byte3 + byte2 + byte1;
}

// 辅助函数：实现16进制字符串按字节反转,用于速度分辨率指令
QString Micromanipulator::swapBytes_2(quint16 data)
{

    // 确保字符串长度是偶数，并且填充为 4 个字符
    QString hexStr = QString::number(data, 16).toUpper().rightJustified(4, '0');


    // 按照两个字符为一组进行拆分
    QString byte1 = hexStr.mid(0, 2);
    QString byte2 = hexStr.mid(2, 2);


    // 按照倒序拼接
    return byte2 + byte1;
}

//设置速度与分辨率
void Micromanipulator::on_BtnSetVR_clicked()
{
    if(mIsOpen == true)
    {
        QString errorMessage;
        const int speed = ui->SetVelocity->text().toInt();
        const QString resolution = ui->SetResolution->currentText();

        if (!m_microArmModule.setVelocityAndResolution(speed, resolution, &errorMessage))
        {
            ui->FeedBack->setText(errorMessage);
            return;
        }

        ui->FeedBack->setText(tr("速度=%1, 分辨率=%2 已下发").arg(speed).arg(resolution));
    }
}

//设置为相对位移模式
void Micromanipulator::on_rBtnRelativeMove_clicked()
{
    if (m_microArmModule.setMoveMode(MicroArm::MoveMode::Relative))
    {
        ui->FeedBack->setText("相对位移模式");
    }
}

//设置为绝对位移模式
void Micromanipulator::on_rBtnAbsoluteMove_clicked()
{
    if (m_microArmModule.setMoveMode(MicroArm::MoveMode::Absolute))
    {
        ui->FeedBack->setText("绝对位移模式");
    }
}

//将当前位置置零
void Micromanipulator::on_BtnSetOrigin_clicked()
{
    if (m_microArmModule.zeroCurrentPosition())
    {
        ui->FeedBack->setText("各轴当前位置置零");
        mic_X = 0;
        mic_Y = 0;
        mic_Z = 0;
    }
}

//打断当前位移指令
void Micromanipulator::on_BtnInterruptMove_clicked()
{
    if (m_microArmModule.interruptMotion())
    {
        ui->FeedBack->setText("发送打断位移指令");
    }
}

//串口发送开关触发（16进制）
void Micromanipulator::on_BtnSerialPortSend_clicked()
{
    if(mIsOpen == true)
    {
        QString SendData = ui->SendData->toPlainText();
        QString errorMessage;
        if (!m_microArmModule.sendManualHexCommand(SendData, &errorMessage))
        {
            ui->FeedBack->setText(errorMessage);
        }
    }
}

//串口接收
void Micromanipulator::SerialReadData(const QByteArray &rawBytes, const QString &hexView)
{
    Q_UNUSED(hexView); // 旧版本提供的字符串视图仅用于调试，现已由字节流解析取代。

    // MicroArm 模块已经完成底层缓冲区读取，此处直接使用其提供的十六进制字符串进行协议解析。
    if (rawBytes.isEmpty()) {
        return; // 如果此次回调没有有效字节，则无需继续处理。
    }

    // Step 1：将原始字节转换为大写十六进制字符串，既方便调试输出，也便于复用原有的字符串解析逻辑。
    const QString compactHexChunk = bytesToUpperHex(rawBytes);
    const QString prettyHexChunk = formatHexForDisplay(compactHexChunk);
    qDebug() << "Received data" << prettyHexChunk;

    accumulatedText += compactHexChunk;

    // Step 2：当正在等待“获取当前位置”回复时，直接在字节层面缓存数据，避免空格或大小写干扰帧长判断。
    if (m_waitingForMicPositionResponse) {
        m_pendingMicPositionBytes.append(rawBytes);

        // 只有在缓冲区累积到至少 13 个字节（12 字节载荷 + 0x0D 结尾）时才尝试解析，
        // 同时从第 12 个索引开始查找结束符，以避免载荷内部本身包含 0x0D 时被误截断。
        if (m_pendingMicPositionBytes.size() >= 13) {
            const int terminatorIndex = m_pendingMicPositionBytes.indexOf('\r', 12);
            if (terminatorIndex != -1) {
                const QByteArray frameBytes = m_pendingMicPositionBytes.left(terminatorIndex + 1);
                handleMicPositionFrame(frameBytes);

                // 清理已经处理完毕的帧数据，并重置等待状态，避免旧数据污染下一次请求。
                m_pendingMicPositionBytes.clear();
                m_waitingForMicPositionResponse = false;
            } else if (m_pendingMicPositionBytes.size() > 256) {
                // 如果累计数据异常增长且仍未检测到结束符，则视为通信异常进行丢弃，防止内存膨胀。
                qWarning() << "Micro-arm frame buffer overflow" << m_pendingMicPositionBytes.size()
                           << "bytes, dropping pending payload";
                m_pendingMicPositionBytes.clear();
                m_waitingForMicPositionResponse = false;
            }
        }
    }
    //机械臂串口指令
    if(accumulatedText.right(2).compare("0D", Qt::CaseInsensitive) == 0 && (accumulatedText.length() == 26 || accumulatedText.length() == 2)){
        ui->ReceiveData->append(formatHexForDisplay(accumulatedText));// 同样可以添加换行符
        if (!m_waitingForMicPositionResponse) {
            MicInfo = accumulatedText;
        }
        accumulatedText.clear();
    }
    //俯仰电机指令
    if(accumulatedText.left(2) == "3E")
    {
        //如果为单圈角度指令（3E94）
        if(accumulatedText.length() == 20 && accumulatedText.left(4) == "3E94")
        {
            QString CMD = accumulatedText.mid(0,8);
            QString CMDSUM = accumulatedText.mid(8,2);

            if(CMDSUM == getSumFromHex(CMD)){
                QString DATA = accumulatedText.mid(10,8);
                QString DATASUM = accumulatedText.mid(18,2);

                if(DATASUM == getSumFromHex(DATA)){
                    ui->ReceiveData->append(accumulatedText);// 同样可以添加换行符
                    QString DATA1 = DATA.mid(0,2);
                    QString DATA2 = DATA.mid(2,2);
                    QString DATA3 = DATA.mid(4,2);
                    QString DATA4 = DATA.mid(6,2);
                    QString DATASwap = DATA4+DATA3+DATA2+DATA1;
                    quint32 DATAfinal = DATASwap.toUInt(nullptr, 16);
                    float angle_f2 = DATAfinal/100.0f;
                    ui->GetSingleLoopAngle->setText(QString::number(angle_f2, 'f', 2));
                    accumulatedText.clear();
                }
                else{
                    qDebug()<<"DATA数据和校验错误";
                    accumulatedText.clear();
                }
            }
            else{
                qDebug()<<"CMD数据和校验错误";
                accumulatedText.clear();
            }
        }

        //如果为多圈角度命令（3E92）
        if(accumulatedText.length() == 28 && accumulatedText.left(4) == "3E92")
        {
            QString CMD = accumulatedText.mid(0,8);
            QString CMDSUM = accumulatedText.mid(8,2);
            if(CMDSUM == getSumFromHex(CMD)){
                QString DATA = accumulatedText.mid(10,16);
                QString DATASUM = accumulatedText.mid(26,2);

                if(DATASUM == getSumFromHex(DATA)){
                    ui->ReceiveData->append(accumulatedText);// 同样可以添加换行符
                    // 翻转字节顺序
                    QString DATASwap;
                    for (int i = 7; i >= 0; --i) { // 8 字节翻转
                        DATASwap += DATA.mid(i * 2, 2);
                    }
                    quint64 DATAfinal = DATASwap.toULongLong(nullptr, 16);
                    float angle_f2 = DATAfinal/100.0f;
                    ui->GetMultiLoopAngle->setText(QString::number(angle_f2, 'f', 2));
                    accumulatedText.clear();
                }
                else{
                    qDebug()<<"DATA数据和校验错误";
                    accumulatedText.clear();
                }
            }
            else{
                qDebug()<<"CMD数据和校验错误";
                accumulatedText.clear();
            }
        }
        //如果为增量式转动指令的回复
        if(accumulatedText.length() == 26 && accumulatedText.left(2) == "3E")
        {
            QString CMD = accumulatedText.mid(0,8);
            QString CMDSUM = accumulatedText.mid(8,2);
            if(CMDSUM == getSumFromHex(CMD)){
                QString DATA = accumulatedText.mid(10,14);
                QString DATASUM = accumulatedText.mid(24,2);

                if(DATASUM == getSumFromHex(DATA)){
                    ui->ReceiveData->append(accumulatedText);// 同样可以添加换行符
                    accumulatedText.clear();
                }
                else{
                    qDebug()<<"DATA数据和校验错误";
                    accumulatedText.clear();
                }
            }
            else{
                qDebug()<<"CMD数据和校验错误";
                accumulatedText.clear();
            }
        }
    }
    //防止其中某一串口接收信息未即时清空
    if(accumulatedText.length() > 35)
    {
        accumulatedText.clear();
    }


}

//===================== 宏观机械臂反馈统一处理 =====================//
// 功能：缓存 MacroArm 模块回传的原始 JSON 字符串，便于按钮查询时直接复用，
//       同时刷新界面显示并输出调试日志，保证调试与 UI 状态一致。
// 输入：text —— 宏观机械臂 TCP 返回的整段文本（可能包含多条报文）。
void Micromanipulator::handleMacroArmFeedback(const QString &text)
{
    // Step 1：记录调试信息，便于和原始日志对照定位通信问题。
    qDebug() << "MacroArm feedback:" << text;

    // Step 2：缓存原始字符串，若按钮触发解析时 UI 尚未刷新，可直接回退到该缓存。
    m_lastMacroArmFeedback = text;

    // Step 3：为界面展示做最小化归一化处理，将回车转换为换行，以免 QTextEdit 出现额外空行。
    QString displayText = text;
    displayText.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    displayText.replace(QChar('\r'), QChar('\n'));
    displayText.replace(QStringLiteral("\\r\\n"), QStringLiteral("\n"));

    // QTextEdit 使用 setPlainText 可避免被误认为富文本，确保与原始内容一致呈现。
    ui->EditReceiveData->setPlainText(displayText);
}

//===================== 十六进制字符串清理与展示辅助 =====================//
// 功能：将字节流直接编码为紧凑的大写十六进制字符串，避免外部空格格式对协议解析造成影响。
QString Micromanipulator::bytesToUpperHex(const QByteArray &bytes) const
{
    return QString::fromLatin1(bytes.toHex().toUpper());
}

// 功能：为了在 UI 中保持易读性，将连续的十六进制串按照两个字符一组重新插入空格。
QString Micromanipulator::formatHexForDisplay(const QString &compactHex) const
{
    QString spaced;
    spaced.reserve(compactHex.length() + compactHex.length() / 2);

    for (int i = 0; i < compactHex.length(); i += 2) {
        if (i > 0) {
            spaced.append(QLatin1Char(' '));
        }
        spaced.append(compactHex.mid(i, 2));
    }

    return spaced;
}

//===================== 解析微动机械臂位置反馈帧 =====================//
// 功能：对长度为 13 字节（12 字节坐标 + 0x0D 结尾）的回复帧进行校验与拆包，提取 XYZ 三轴坐标。
// 输入：frameBytes —— 含有结束符 0x0D 的完整返回帧，例如 {0x70,0x11,...,0x0D}。
// 步骤：
//   1) 确认帧至少包含 13 字节且最后一字节为 0x0D。
//   2) 拆取前三个 4 字节字段，按小端序解码为整型坐标。
//   3) 将原始帧转换为十六进制字符串缓存，更新 MicInfo 并刷新 UI。
void Micromanipulator::handleMicPositionFrame(const QByteArray &frameBytes)
{
    constexpr int kPayloadBytes = 12;          // XYZ 共计 12 字节。
    constexpr unsigned char kFrameTerminator = 0x0D; // 协议定义的结尾符。

    // Step 1：基础协议校验——检查字节长度以及结束符是否为 0x0D。
    if (frameBytes.size() < kPayloadBytes + 1 ||
        static_cast<unsigned char>(frameBytes.back()) != kFrameTerminator) {
        qWarning() << "Invalid micro-arm frame (size =" << frameBytes.size() << ")"
                   << formatHexForDisplay(bytesToUpperHex(frameBytes));
        return;
    }

    if (frameBytes.size() != kPayloadBytes + 1) {
        qWarning() << "Unexpected micro-arm frame length:" << frameBytes.size()
                   << "payload:" << formatHexForDisplay(bytesToUpperHex(frameBytes));
        // 若长度异常仍尝试继续解析，但记录告警方便排查硬件通信。
    }

    // Step 2：解析小端序坐标值。frameBytes[0..3] 为 X， [4..7] 为 Y， [8..11] 为 Z。
    auto decodeLittleEndian = [](const QByteArray &bytes, int startIndex) {
        int value = 0;
        for (int offset = 3; offset >= 0; --offset) {
            value = (value << 8) | static_cast<unsigned char>(bytes[startIndex + offset]);
        }
        return value;
    };

    const QByteArray payloadBytes = frameBytes.left(kPayloadBytes);

    mic_X = decodeLittleEndian(payloadBytes, 0);
    mic_Y = decodeLittleEndian(payloadBytes, 4);
    mic_Z = decodeLittleEndian(payloadBytes, 8);

    // Step 3：记录原始帧并更新 UI 显示。
    MicInfo = bytesToUpperHex(frameBytes);
    ui->FeedBack->setText(QStringLiteral("Mic Position -> X:%1 Y:%2 Z:%3")
                              .arg(mic_X)
                              .arg(mic_Y)
                              .arg(mic_Z));
}

//同步更新速度输入框
void Micromanipulator::updateLineEdit(int value)
{
    ui->SetVelocity->setText(QString::number(value));
    // float Stepspersec = roundToTwoDecimals(ui->SetVelocity->text());
    // float Resolution = roundToTwoDecimals(ui->SetResolution->currentText());


    // float Speed = Stepspersec * Resolution;
    // QString speedString = QString::number(Speed,'f',2);
    // ui->ShowVelocity->setText(speedString);
}

//更新滑动条
void Micromanipulator::updateSliderFromText()
{
    int Speedrange;

    QString resolution = ui->SetResolution->currentText();
    if(resolution == "0.2")
    {
        Speedrange = 6550;
    }
    else
    {
        Speedrange = 1310;
    }
    ui->VelocitySlider->setRange(0,Speedrange);
    ui->VelocitySlider->setSingleStep(1);
    bool ok;
    int value = ui->SetVelocity->text().toInt(&ok);
    if (ok && value >= 0 && value <= Speedrange) {
        ui->VelocitySlider->setValue(value);

    } else {
        // 如果输入无效，可以重置为当前滑块的值或做其他处理
        ui->SetVelocity->setText(QString::number(ui->VelocitySlider->value()));
    }
}

// 辅助函数：将QString转换为float并舍入到两位小数
float Micromanipulator::roundToTwoDecimals(const QString& text) {
    bool ok;
    float value = text.toFloat(&ok);
    if (!ok) {
        // 处理转换失败的情况，这里简单返回0.0
        return 0.0f;
    }
    // 乘以100，四舍五入，再除以100
    return std::round(value * 100.0f) / 100.0f;
}

//执行求解变换矩阵
void Micromanipulator::on_BtnGetTrans_clicked()
{
    GetTrans(trans_x,trans_y);
    camera_x1 = VisualPosition_X;
    camera_y1 = VisualPosition_Y;
    qDebug()<<camera_x1<<camera_y1;

    GetTrans(trans_x,0);
    camera_x2 = VisualPosition_X;
    camera_y2 = VisualPosition_Y;
    qDebug()<<camera_x2<<camera_y2;

    GetTrans(0,trans_y);
    camera_x3 = VisualPosition_X;
    camera_y3 = VisualPosition_Y;

    qDebug()<<camera_x2<<camera_y2;

    cv::Mat A = (cv::Mat_<int>(6, 6) <<
                 camera_x1, camera_y1, 1, 0, 0, 0,
                 0, 0, 0, camera_x1, camera_y1, 1,
                 camera_x2, camera_y2, 1, 0, 0, 0,
                 0, 0, 0, camera_x2, camera_y2, 1,
                 camera_x3, camera_y3, 1, 0, 0, 0,
                 0, 0, 0, camera_x3, camera_y3, 1);

    cv::Mat B = (cv::Mat_<int>(6, 1) << trans_x, trans_y, trans_x, 0, 0, trans_y);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            std::cout << A.at<int>(i, j) << " ";
        }
        std::cout <<std::endl;
    }

    cv::Mat A_float;
    cv::Mat B_float;
    A.convertTo(A_float, CV_32F);
    B.convertTo(B_float, CV_32F);
    // 转换为32位浮点数矩阵

    // 计算矩阵A的行列式，以检查其是否是奇异矩阵
    double detA = cv::determinant(A_float);

    // 检查行列式是否接近零（使用一个小的阈值来避免浮点误差）
    if (std::abs(detA) < 1e-10) {
        qDebug()<<"奇异矩阵无解";
        return;
    }
    else{

        // 使用cv::solve函数来求解方程组Ax=B
        cv::Mat x;
        cv::solve(A_float, B_float, x);
        a = x.at<float>(0, 0);
        b = x.at<float>(1, 0);
        c = x.at<float>(2, 0);
        d = x.at<float>(3, 0);
        e = x.at<float>(4, 0);
        f = x.at<float>(5, 0);
        qDebug()<<a<<b<<c<<d<<e<<f;
        // 变换矩阵获取状态置1
        TransMatGet = true;



    }
}

//矩阵求解过程
void Micromanipulator::GetTrans(qint32 x, qint32 y)
{


    //根据函数获取的xy进行移动（z不移动）
    qint32 z = 0;
    if (m_microArmModule.moveToPose(x, y, z))
    {
        mic_X = x;
        mic_Y = y;
        mic_Z = z;
    }

    QTime dieTime = QTime::currentTime().addMSecs(1500);

    while( QTime::currentTime() < dieTime )
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    }


}

//移动置指定位置（增量式）
void Micromanipulator::on_BtnAngleTrans_clicked()
{
    bool ok;
    double MoveAngleDouble = ui->SetAlphaTransition->text().toDouble(&ok);
    if(!ok)
    {
        ui->FeedBack->setText("旋转角度格式有误，重新设置");
    }
    else
    {
        // 乘以1000，然后截断小数部分（通过类型转换）
        qint32 MoveAngleInt = static_cast<int>(MoveAngleDouble * 1000);

        QString MoveAngleHex = swapBytes_1(MoveAngleInt);
        qDebug()<<MoveAngleHex;
        QString checkHex = getSumFromHex(MoveAngleHex);
        QString TransAngle = "3EA70104EA" + MoveAngleHex + checkHex;
        QByteArray Data_final = QByteArray::fromHex(TransAngle.toUtf8());
        qDebug()<<"转轴移动指令"<<TransAngle;
        mSerialPort_2.write(Data_final);


    }
}

//辅助函数：用于求解旋转电机的校验位
QString Micromanipulator::getSumFromHex(QString HexString)
{

    qint32 HexSum = 0;
    // 遍历字符串的每两个字符
    for (int i = 0; i < HexString.size(); i += 2) {
        // 提取两个字符组成的子字符串
        QString hexByte = HexString.mid(i, 2);

        // 将16进制字符串转换为整数
        bool ok;
        uint8_t hexValue = hexByte.toUInt(&ok, 16);
        if (ok) {
            HexSum += hexValue;
        } else {
            qWarning() << "16进制转整数失败" << hexByte;
        }
    }
    // 将HexSum转换为十六进制字符串，并确保是大写
    QString hexSumString = QString("%1").arg(HexSum,2 ,16, QChar('0')).toUpper().right(2);
        // QString::number(HexSum, 16).toUpper().right(2);

    return hexSumString;
}

// 获取单圈角度信息
void Micromanipulator::on_BtnGetSingleLoopAngle_clicked()
{
    QString GetSingleLoopAngle = "3E940100D3";

    QByteArray SingleLoop = QByteArray::fromHex(GetSingleLoopAngle.toUtf8());

    mSerialPort_2.write(SingleLoop);

    Delay(100);

    QByteArray data = mSerialPort_2.readAll();

    if (!data.isEmpty()) {
        QString text = data.toHex().toUpper();

        QString CMD = text.mid(0,8);
        QString CMDSUM = text.mid(8,2);

        if(CMDSUM == getSumFromHex(CMD)){
            QString DATA = text.mid(10,8);
            QString DATASUM = text.mid(18,2);

            if(DATASUM == getSumFromHex(DATA)){
                QString DATA1 = DATA.mid(0,2);
                QString DATA2 = DATA.mid(2,2);
                QString DATA3 = DATA.mid(4,2);
                QString DATA4 = DATA.mid(6,2);
                QString DATASwap = DATA4+DATA3+DATA2+DATA1;
                quint32 DATAfinal = DATASwap.toUInt(nullptr, 16);
                float angle_f2 = DATAfinal/1000.0f;
                ui->GetSingleLoopAngle->setText(QString::number(angle_f2, 'f', 3));
            }
            else{
                qDebug()<<"DATA数据和校验错误";
            }
        }
        else{
            qDebug()<<"CMD数据和校验错误";
        }
    }

}

//将当前位置作为电机零位
void Micromanipulator::on_BtnInterrupt_clicked()
{
    QString Interrupt = "3E810100C0";

    QByteArray Message = QByteArray::fromHex(Interrupt.toUtf8());

    mSerialPort_2.write(Message);
}

// 获取多圈角度信息
void Micromanipulator::on_BtnGetMultiLoopAngle_clicked()
{
    QString GetMultiLoopAngle = "3E920100D1";

    QByteArray MultiLoop = QByteArray::fromHex(GetMultiLoopAngle.toUtf8());

    mSerialPort_2.write(MultiLoop);

    Delay(100);

    QByteArray data = mSerialPort_2.readAll();

    if (!data.isEmpty()) {
        QString text = data.toHex().toUpper();

        QString CMD = text.mid(0,8);
        QString CMDSUM = text.mid(8,2);
        if(CMDSUM == getSumFromHex(CMD)){
            QString DATA = text.mid(10,16);
            QString DATASUM = text.mid(26,2);

            if(DATASUM == getSumFromHex(DATA)){
                QString DATASwap;
                for (int i = 7; i >= 0; --i) { // 8 字节翻转
                    DATASwap += DATA.mid(i * 2, 2);
                }
                qint64 DATAfinal = DATASwap.toULongLong(nullptr, 16);
                float angle_f2 = DATAfinal/1000.0f;
                ui->GetMultiLoopAngle->setText(QString::number(angle_f2, 'f', 3));
            }
            else{
                qDebug()<<"DATA数据和校验错误";
            }
        }
        else{
            qDebug()<<"CMD数据和校验错误";
        }
    }
}

//移动至零角
void Micromanipulator::on_BtnMoveToZero_clicked()
{
    QString ZeroAngle = "3EA30108EA000000000000000000";
    QByteArray GotoZeroAngle = QByteArray::fromHex(ZeroAngle.toUtf8());
    mSerialPort_2.write(GotoZeroAngle);

}

//转轴电机参数调节
void Micromanipulator::on_SaveAnglePIDToRAM_clicked()
{
    bool ok , ok2 , ok3;
    int Kp = ui->SetAngleKp->text().toInt(&ok);
    int Ki = ui->SetAngleKi->text().toInt(&ok2);
    int Kd = ui->SetAngleKd->text().toInt(&ok3);
    if(!ok || !ok2 || ok3){
        ui->FeedBack->setText("数据类型错误");
    }
    if(Kp<0||Kp>2000 || Ki<0||Ki>2000 || Kd<0||Kd>2000){
        ui->FeedBack->setText("超出PID参数范围（0~2000）");
    }else{
        QString KpCommand = swapBytes_2(Kp);
        QString KiCommand = swapBytes_2(Ki);
        QString KdCommand = swapBytes_2(Kd);
        QString KpidCommand ="96" + KpCommand + KiCommand + KdCommand;
        QString Sum = getSumFromHex(KpidCommand);
        QString PIDCommand = "3E4201078896" + KpCommand+ KiCommand + KdCommand +Sum;
        QByteArray Data_final = QByteArray::fromHex(PIDCommand.toUtf8());
        mSerialPort_2.write(Data_final);
        ui->FeedBack->setText("位置环PID参数保存");
    }
}

void Micromanipulator::on_SaveSpeedPIDToRAM_clicked()
{
    bool ok , ok2 , ok3;
    int Kp = ui->SetSpeedKp->text().toInt(&ok);
    int Ki = ui->SetSpeedKi->text().toInt(&ok2);
    int Kd = ui->SetSpeedKd->text().toInt(&ok3);
    if(!ok || !ok2 || ok3){
        ui->FeedBack->setText("数据类型错误");
    }
    if(Kp<0||Kp>2000 || Ki<0||Ki>2000 || Kd<0||Kd>2000){
        ui->FeedBack->setText("超出PID参数范围（0~2000）");
    }else{
        QString KpCommand = swapBytes_2(Kp);
        QString KiCommand = swapBytes_2(Ki);
        QString KdCommand = swapBytes_2(Kd);
        QString KpidCommand = "97" + KpCommand + KiCommand + KdCommand;
        QString Sum = getSumFromHex(KpidCommand);
        QString PIDCommand = "3E4201078897" + KpCommand+ KiCommand + KdCommand +Sum;
        QByteArray Data_final = QByteArray::fromHex(PIDCommand.toUtf8());
        mSerialPort_2.write(Data_final);
        ui->FeedBack->setText("速度环PID参数保存");
    }
}

void Micromanipulator::on_SavePIDToROM_clicked()
{

}

//当前图像截图
void Micromanipulator::on_BtnGetFrame_clicked()
{
    static int saveCount = 0; // 或者放在类里做成员变量更好
    std::string filename = "C:/Users/Lee/Desktop/BiaoDing/ScreenShot_" + std::to_string(saveCount++) + ".jpg";
    const cv::Mat &source = m_lastDisplayedFrame.empty() ? frame : m_lastDisplayedFrame;
    if (source.empty()) {
        QMessageBox::warning(this, tr("提示"), tr("当前没有可保存的画面"));
        return;
    }

    cv::imwrite(filename, source);
    // cv::imwrite("C:/Users/Lee/Desktop/BiaoDing.jpg",frame);
}


void Micromanipulator::Delay(int Times_ms){
    QTime dieTime = QTime::currentTime().addMSecs(Times_ms);
    while( QTime::currentTime() < dieTime )
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    }
}

void Micromanipulator::on_BtnGetParameter_u_clicked(){
    //求解缩放系数u
    //初始机械臂末端坐标
    int robotx1 = VisualPosition_X;
    int roboty1 = VisualPosition_Y;
    //机械臂x轴移动4000步
    qint32 x = 50000;
    qint32 y = 0;
    qint32 z = 0;
    if (m_microArmModule.moveToPose(x, y, z))
    {
        mic_X = x;
        mic_Y = y;
        mic_Z = z;
    }
    Delay(5000);
    int robotx2 = VisualPosition_X;
    int roboty2 = VisualPosition_Y;
    // 计算像素距离
    double Imagedistance = std::sqrt(std::pow(robotx2 - robotx1, 2) + std::pow(roboty2 - roboty1, 2));
    // 缩放系数u
    Parameter_u = (x * 0.04) / (1000 * Imagedistance);
    qDebug()<<"distance"<<Imagedistance;
    qDebug()<<"u="<<Parameter_u;
}

// 求解旋转轴上的注射针长度
void Micromanipulator::on_BtnGetTipLength_clicked()
{
    // int x0 = ServoPosition_X;
    // QString Angle_1 = "3EA70104EAC8000000C8";//逆转2度
    // QString Angle_2 = "3EA70104EA70FEFFFF6C";//正转4度
    // QByteArray AngleFront = QByteArray::fromHex(Angle_1.toUtf8());
    // QByteArray AngleBack = QByteArray::fromHex(Angle_2.toUtf8());

    // //逆时针转2度
    // m_microArmModule.sendRawCommand(AngleFront);
    // Delay(4000);
    // int x1 = ServoPosition_X;

    // //顺时针转4度
    // m_microArmModule.sendRawCommand(AngleBack);
    // Delay(4000);
    // int x2 = ServoPosition_X;

    // //角度归位
    // m_microArmModule.sendRawCommand(AngleFront);

    // //求解dertx、cos、sin
    // int dertX_f = x1-x0;
    // int dertX_b = x2-x0;
    // double CosAlpha = cos(2 * M_PI / 180.0);
    // double SinAlpha = sin(2 * M_PI / 180.0);



    // double xn0 = Parameter_u*(dertX_f+dertX_b)/(2*CosAlpha-2);
    // double zn0 = Parameter_u*(dertX_f-dertX_b)/(2*SinAlpha);
    // double result = std::sqrt(std::pow(xn0, 2) + std::pow(zn0, 2));
    // qDebug()<<"u="<<Parameter_u
    //          <<"dertx_f="<<dertX_f
    //          <<"dertx_b="<<dertX_b
    //          <<"CosAlpha="<<CosAlpha
    //          <<"SinAlpha="<<SinAlpha
    //          <<"xn0="<<xn0
    //          <<"zn0="<<zn0
    //          <<"TipLength="<<result;

    // 测试用
    x0 = VisualPosition_X;
    QString Angle_1 = "3EA70104EA881300009B";//逆转5度
    QByteArray AngleFront = QByteArray::fromHex(Angle_1.toUtf8());
    mSerialPort_2.write(AngleFront);



}

void Micromanipulator::on_BtnGetTipLength_2_clicked()
{
    x1 = VisualPosition_X;
    QString Angle_2 = "3EA70104EAF0D8FFFFC6";//正转10度
    QByteArray AngleBack = QByteArray::fromHex(Angle_2.toUtf8());
    mSerialPort_2.write(AngleBack);
}

void Micromanipulator::on_BtnGetTipLength_3_clicked()
{
    x2 = VisualPosition_X;
    QString Angle_1 = "3EA70104EA881300009B";//逆转5度
    QByteArray AngleFront = QByteArray::fromHex(Angle_1.toUtf8());
    mSerialPort_2.write(AngleFront);

    int dertX_f = x1-x0;
    int dertX_b = x2-x0;
    double CosAlpha = cos(5.0 * M_PI / 180.0);
    double SinAlpha = sin(5.0 * M_PI / 180.0);
    // double xn0 = Parameter_u*(dertX_f+dertX_b)/(2*CosAlpha-2);
    // double zn0 = Parameter_u*(dertX_f-dertX_b)/(2*SinAlpha);
    double a11 = Matrix_Camera_Mic.at<double>(0,0);
    double a13 = Matrix_Camera_Mic.at<double>(0,2);
    double xn0 = Parameter_u*(a11*(CosAlpha-1)*(dertX_f-dertX_b)-a13*SinAlpha*(dertX_f+dertX_b))/(2*SinAlpha*(CosAlpha-1)*(std::pow(a11, 2) + std::pow(a13, 2)));
    double zn0 = Parameter_u*(a13*(CosAlpha-1)*(dertX_f-dertX_b)+a11*SinAlpha*(dertX_f+dertX_b))/(2*SinAlpha*(CosAlpha-1)*(std::pow(a11, 2) + std::pow(a13, 2)));

    Ltip = std::sqrt(std::pow(xn0, 2) + std::pow(zn0, 2));
    qDebug()<<"u="<<Parameter_u
             <<"dertx_f="<<dertX_f
             <<"dertx_b="<<dertX_b
             <<"CosAlpha="<<CosAlpha
             <<"SinAlpha="<<SinAlpha
             <<"xn0="<<xn0
             <<"zn0="<<zn0
             <<"TipLength="<<Ltip;
}

//与俯仰串口相关
//串口参数初始化
bool Micromanipulator::getSerialPortConfig_2()
{
    //获取串口配置
    mPortName_2 = ui->CboxSerialPort_2->currentText();
    mBaudRate_2 = ui->CboxBaudRate_2->currentText();

    //设置串口配置
    mSerialPort_2.setPortName(mPortName_2);
    //设置波特率
    if(ui->CboxBaudRate_2->currentText() == "1200")
    {
        mSerialPort_2.setBaudRate(QSerialPort::Baud1200);
    }
    else if(ui->CboxBaudRate_2->currentText() == "2400")
    {
        mSerialPort_2.setBaudRate(QSerialPort::Baud2400);
    }
    else if(ui->CboxBaudRate_2->currentText() == "4800")
    {
        mSerialPort_2.setBaudRate(QSerialPort::Baud4800);
    }
    else if(ui->CboxBaudRate_2->currentText() == "9600")
    {
        mSerialPort_2.setBaudRate(QSerialPort::Baud9600);
    }
    else if(ui->CboxBaudRate_2->currentText() == "19200")
    {
        mSerialPort_2.setBaudRate(QSerialPort::Baud19200);
    }
    else if(ui->CboxBaudRate_2->currentText() == "115200")
    {
        mSerialPort_2.setBaudRate(QSerialPort::Baud115200);
    }

    if (mSerialPort_2.open(QSerialPort::ReadWrite))  {
        return true;
    } else {
        // 处理打开串口失败的情况
        ui->FeedBack->setText("串口"+mPortName_2+"打开失败");
        return false;
    }


}

//串口开关触发
void Micromanipulator::on_BtnSerialPortOnOff_2_clicked()
{
    if(mIsOpen_2 == true)
    {
        mSerialPort_2.close();
        ui->BtnSerialPortOnOff_2->setText("打开");
        mIsOpen_2 = false;
        ui->CboxSerialPort_2->setEnabled(true);
        ui->CboxBaudRate_2->setEnabled(true);
        ui->FeedBack->clear();
        ui->FeedBack->append("串口已关闭");
    }
    else
    {
        if(getSerialPortConfig_2() == true)
        {
            mIsOpen_2 = true;
            ui->BtnSerialPortOnOff_2->setText("断开");
            qDebug() << "成功打开串口"<<mPortName_2;
            ui->FeedBack->append("成功打开串口"+mPortName_2);
            // 将串口参数选项禁用
            ui->CboxSerialPort_2->setEnabled(false);
            ui->CboxBaudRate_2->setEnabled(false);
        }
        else
        {
            mIsOpen_2 = false;
        }
    }
}


void Micromanipulator::on_BtnReadCurrentAngle_clicked()
{
    if(mIsOpen_2 == true)
    {
        QByteArray CurrentPosition = QByteArray::fromHex("3E940100D3");
        ui->FeedBack->setText("读取角度");
        mSerialPort_2.write(CurrentPosition);
        mSerialPort_2.readAll();
        Delay(500);

        QByteArray receivedData = mSerialPort_2.readAll();

        if (!receivedData.isEmpty())  {
            // 将接收到的数据转换为十六进制字符串

            QByteArray extractedData = receivedData.mid(5,  4);
            QString hexString = extractedData.toHex();

            // 按两个字符拆分并倒序拼接
            QString reversedHex = hexString.mid(6,2)  +
                                  hexString.mid(4,2)  +
                                  hexString.mid(2,2)  +
                                  hexString.mid(0,2);
            bool ok;
            qint32 decimalValue = reversedHex.toInt(&ok,  16);  // 十六进制转十进制整数
            if (ok) {
                double result = decimalValue / 1000.0;          // 除以100保留两位小数
                double theta = ui->AssistAngle->text().toDouble();
                double Alpha = result - theta;

                CurrentAngle = Alpha;
                ui->CurrentAngle->setText(QString::number(Alpha, 'f', 3));
            } else {
                ui->FeedBack->setText("角度转换失败");
            }

        }else{
            ui->FeedBack->setText("接收信息为空");
        }
    }
}



//结合已知角度、目标角度实现定心转动
void Micromanipulator::on_BtnRCM_clicked()
{
    bool ok1, ok2;
    ui->BtnReadCurrentAngle->click();
    Delay(500);
    QString CurrentAngle = ui->CurrentAngle->text();
    QString TargetAngle = ui->TargetAngle->text();
    double CAngle = CurrentAngle.toDouble(&ok1);   // 通过ok1判断是否转换成功
    double TAngle = TargetAngle.toDouble(&ok2);

    // 校验转换结果
    if (!ok1 || !ok2) {
        return;
    }

    ui->BtnGetCurrentState->click();
    Delay(1000);

    cv::Vec6i macroPose{
        ui->XPosition->text().toInt(),
        ui->YPosition->text().toInt(),
        ui->ZPosition->text().toInt(),
        ui->RXPosition->text().toInt(),
        ui->RYPosition->text().toInt(),
        ui->RZPosition->text().toInt()
    };

    QString errorMessage;
    if (!m_macroMicroController.executeRCMPitchAlignment(CAngle, TAngle, 130.0, macroPose, &errorMessage)) {
        if (!errorMessage.isEmpty()) {
            QMessageBox::warning(this, tr("警告"), errorMessage, QMessageBox::Ok);
        }
    }
}

//socket

void Micromanipulator::on_BtnSocketConnect_clicked()
{
    if(mSocketIsOpen == 0){
        QString ip_01 = ui->IPAdress1->text();
        QString ip_02 = ui->IPAdress2->text();
        QString ip_03 = ui->IPAdress3->text();
        QString ip_04 = ui->IPAdress4->text();
        int SocketPortNum1 = ui->SocketPortNum1->text().toInt();
        // const int SocketPortNum2 = ui->SocketPortNum2->text().toInt(); // ⚠️ 反馈端口暂未启用。
        const QString ip = ip_01+"."+ip_02+"."+ip_03+"."+ip_04;
        QString connectError;
        if (!m_macroArmModule.configureAndConnect(ip,
                                                  static_cast<quint16>(SocketPortNum1),
                                                  &connectError))
        {
            ui->FeedBack->setText(connectError);
            return;
        }
        m_RobotSocket->connectToHost(ip, SocketPortNum1);
        // if (m_feedbackSocket) {
        //     m_feedbackSocket->connectToHost(ip, SocketPortNum2);
        // }
        ui->IPAdress1->setDisabled(1);
        ui->IPAdress2->setDisabled(1);
        ui->IPAdress3->setDisabled(1);
        ui->IPAdress4->setDisabled(1);
        ui->SocketPortNum1->setDisabled(1);
        ui->SocketPortNum2->setDisabled(1);
        ui->BtnSocketConnect->setText("断开");
        ui->CBoxSelectSocket->addItem(ui->SocketPortNum1->text());
        // ui->CBoxSelectSocket->addItem(ui->SocketPortNum2->text()); // ⚠️ 暂停 8089 反馈端口选择。
        mSocketIsOpen = 1;
    }
    else if(mSocketIsOpen == 1 ){
        m_RobotSocket->abort();
        if (m_feedbackSocket) {
            m_feedbackSocket->abort();
        }
        ui->IPAdress1->setDisabled(0);
        ui->IPAdress2->setDisabled(0);
        ui->IPAdress3->setDisabled(0);
        ui->IPAdress4->setDisabled(0);
        ui->SocketPortNum1->setDisabled(0);
        ui->SocketPortNum2->setDisabled(0);
        ui->BtnSocketConnect->setText("连接");
        ui->CBoxSelectSocket->clear();
        mSocketIsOpen = 0;
        m_macroArmModule.disconnectFromRobot();
    }
}

void Micromanipulator::TcpSocketSend(QTcpSocket *SelectedSocket,QString Message)
{
    if (SelectedSocket == m_RobotSocket) {
        // 同步调用 MacroArm 模块，确保宏观机械臂相关逻辑集中管理。
        m_macroArmModule.sendCommand(Message);
        SelectedSocket->write(Message.toUtf8());
        SelectedSocket->flush();
        return;
    }

    // ====== ⚠️ 反馈端口逻辑已停用：避免误向未启用的 socket 写入 ======
    if (SelectedSocket == m_feedbackSocket) {
        qWarning() << "TcpSocketSend -> 反馈端口已停用，忽略写入请求";
        return;
    }
}


void Micromanipulator::on_BtnSendMessage_clicked()
{
    QString Message = ui->EditSendMessage->toPlainText();
    if(ui->CBoxSelectSocket->currentText()==ui->SocketPortNum1->text()){
        if(m_RobotSocket){
            TcpSocketSend(m_RobotSocket,Message);
        }
    }
}

// 新增的槽函数，用于读取接收到的数据
void Micromanipulator::readSocketData()
{
    QTcpSocket *socket = qobject_cast<QTcpSocket*>(sender());
    if (socket) {
        if (socket == m_RobotSocket) {
            QByteArray data = socket->readAll();
            QString message = QString::fromUtf8(data);
            // 由于界面文本在稍后会被解析为 JSON，这里同步缓存并做最小化格式修正。
            m_lastMacroArmFeedback = message;

            QString displayText = message;
            displayText.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
            displayText.replace(QChar('\r'), QChar('\n'));
            displayText.replace(QStringLiteral("\\r\\n"), QStringLiteral("\n"));
            ui->EditReceiveData->setPlainText(displayText);
        }

    }

}

void Micromanipulator::on_CheckBoxEnableRobot_clicked()
{
    if(ui->CheckBoxEnableRobot->isChecked()){
        m_macroArmModule.setArmPower(true);
    }else{
        m_macroArmModule.setArmPower(false);
    }
}


void Micromanipulator::on_BtnClearError_clicked()
{

}


void Micromanipulator::on_BtnSetAccL_clicked()
{
    const int acceleration = ui->SetAccL->text().toInt();
    m_macroArmModule.setMaxLineAcceleration(acceleration);
}


void Micromanipulator::on_BtnSetSpeedL_clicked()
{
    const int speed = ui->SetSpeedL->text().toInt();
    m_macroArmModule.setMaxLineSpeed(speed);
}

void Micromanipulator::on_BtnGetCurrentState_clicked()
{
    connect(m_JsonExplain, &JsonExplain::stateParsed, this, [=](const QString &state){
        ui->FeedBack->setText(state);
    });

    connect(m_JsonExplain, &JsonExplain::jointAnglesParsed, this, [=](const QList<int> &angles){
        if (angles.size() > 0) ui->J1Angle->setText(QString::number(angles[0]));
        if (angles.size() > 1) ui->J2Angle->setText(QString::number(angles[1]));
        if (angles.size() > 2) ui->J3Angle->setText(QString::number(angles[2]));
        if (angles.size() > 3) ui->J4Angle->setText(QString::number(angles[3]));
        if (angles.size() > 4) ui->J5Angle->setText(QString::number(angles[4]));
        if (angles.size() > 5) ui->J6Angle->setText(QString::number(angles[5]));
    });

    connect(m_JsonExplain, &JsonExplain::poseParsed, this, [=](const QList<int> &pose){
        if (pose.size() > 0) ui->XPosition->setText(QString::number(pose[0]));
        if (pose.size() > 1) ui->YPosition->setText(QString::number(pose[1]));
        if (pose.size() > 2) ui->ZPosition->setText(QString::number(pose[2]));
        if (pose.size() > 3) ui->RXPosition->setText(QString::number(pose[3]));
        if (pose.size() > 4) ui->RYPosition->setText(QString::number(pose[4]));
        if (pose.size() > 5) ui->RZPosition->setText(QString::number(pose[5]));
        curX = pose[0];
        curY = pose[1];
        curZ = pose[2];
        curRx = pose[3];
        curRy = pose[4];
        curRz = pose[5];
    });



    m_macroArmModule.requestCurrentState();
    // 等待数据返回
    Delay(30);
    QString messageReceived = ui->EditReceiveData->toPlainText();
    if (messageReceived.trimmed().isEmpty()) {
        // 若界面尚未来得及刷新，则回退到最新一次缓存的原始反馈，避免出现空串解析失败。
        messageReceived = m_lastMacroArmFeedback;
    }
    m_JsonExplain->JsonGetCurrentState(messageReceived);
}

void Micromanipulator::on_BtnMoveJ_clicked()
{

}

void Micromanipulator::on_BtnMoveL_clicked()
{
    int x  = ui->XPosition->text().toInt();
    int y  = ui->YPosition->text().toInt();
    int z  = ui->ZPosition->text().toInt();
    int rx = ui->RXPosition->text().toInt();
    int ry = ui->RYPosition->text().toInt();
    int rz = ui->RZPosition->text().toInt();
    int v = ui->velocityRatio->text().toInt();
    // === 宏观机械臂指令整合：统一由 MacroArm 模块下发 MoveL ===
    m_macroArmModule.moveL(x, y, z, rx, ry, rz, v);
}

void Micromanipulator::on_BtnMoveC_clicked()
{
    int x  = ui->XPosition->text().toInt();
    int y  = ui->YPosition->text().toInt();
    int z  = ui->ZPosition->text().toInt();
    int rx = ui->RXPosition->text().toInt();
    int ry = ui->RYPosition->text().toInt();
    int rz = ui->RZPosition->text().toInt();
    int via_x  = ui->XPosition_v->text().toInt();
    int via_y  = ui->YPosition_v->text().toInt();
    int via_z  = ui->ZPosition_v->text().toInt();
    int via_rx = ui->RXPosition_v->text().toInt();
    int via_ry = ui->RYPosition_v->text().toInt();
    int via_rz = ui->RZPosition_v->text().toInt();
    int v = ui->velocityRatio->text().toInt();
    // === 宏观机械臂指令整合：统一由 MacroArm 模块下发 MoveC ===
    m_macroArmModule.moveC(via_x, via_y, via_z,
                           via_rx, via_ry, via_rz,
                           x, y, z, rx, ry, rz,
                           v,
                           /*blendRadius*/ 0,
                           /*loop*/ 0,
                           /*trajectory_connect*/ false);
}

void Micromanipulator::on_Btn_RobotStop_clicked()
{
    m_macroArmModule.requestArmStop(false);
    mStop = 1;
    m_macroMicroController.requestStop();
}

void Micromanipulator::on_Btn_RobotSlowStop_clicked()
{
    m_macroArmModule.requestArmStop(true);
}

void Micromanipulator::on_Btn_calibration_MicRobot_clicked()
{
    //当前位置至于零点
    ui->BtnSetOrigin->click();

    // 参数初始化
    cv::Size board_size(4,3);          // 棋盘格内角点数(行,列)
    int square_size = 25000;          // 棋盘格单格(机械臂坐标系下）

    std::vector<cv::Point2f> arm_pts;
    std::vector<cv::Point2f> img_pts;

    for(int i=0; i<board_size.height;  i++)
    {
        for(int j=0; j<board_size.width;  j++)
        {
            // 构造坐标
            arm_pts.emplace_back(j * square_size, i * square_size);

            // 控制机器人移动到对应位置
            if (m_microArmModule.moveToPose(j * square_size, i * square_size, 0))
            {
                mic_X = j * square_size;
                mic_Y = i * square_size;
                mic_Z = 0;
            }

            // 使用 QEventLoop 等待“捕捉”按钮
            QEventLoop loop;
            // 当点击 ui->BtnCapture 时，退出事件循环
            connect(ui->Btn_Capture_MicRobot, &QPushButton::clicked, &loop, &QEventLoop::quit);
            loop.exec(); // 阻塞等待按钮点击

            // 按钮点击后，视觉返回当前图像中的像素位置
            img_pts.emplace_back(VisualPosition_X, VisualPosition_Y);

        }
    }
    H_calib = cv::findHomography(img_pts, arm_pts, cv::RANSAC);
    H_calib.convertTo(H_calib, CV_64F);
    H1 = H_calib;
    H1_linear = cv::Matx22d(H_calib.at<double>(0,0), H_calib.at<double>(0,1),
                            H_calib.at<double>(1,0), H_calib.at<double>(1,1));
    resetOnlineUpdateState();
    ui->BtnGetCurrentPosition->click();
    Delay(500);
    alpha_calib_rad = curRz/1000.0;
    qDebug()<<"标定时的角度为"<<alpha_calib_rad;
    qDebug()<<H_calib.at<double>(0,0)<<H_calib.at<double>(0,1)<<H_calib.at<double>(0,2)
           <<H_calib.at<double>(1,0)<<H_calib.at<double>(1,1)<<H_calib.at<double>(1,2)
           <<H_calib.at<double>(2,0)<<H_calib.at<double>(2,1)<<H_calib.at<double>(2,2);

}

void Micromanipulator::drawCornerPoints(const cv::Mat& src_img,
                                        const std::vector<std::vector<cv::Point2f>>& img_points) {
    // 新建Mat（尺寸与输入图像一致）
    cv::Mat canvas;

    canvas = cv::Mat::zeros(src_img.size(),  CV_8UC3);
    canvas.setTo(cv::Scalar(255,  255, 255)); // 白色背景


    // 定义颜色和标记
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),   // 红
        cv::Scalar(0, 255, 0),   // 绿
        cv::Scalar(255, 0, 0),   // 蓝
        cv::Scalar(0, 255, 255), // 黄
        cv::Scalar(255, 0, 255)  // 紫
    };

    // 绘制角点
    for (size_t i = 0; i < img_points.size();  ++i) {
        const auto& points = img_points[i];
        cv::Scalar color = colors[i % colors.size()];

        for (size_t j = 0; j < points.size();  ++j) {
            const cv::Point2f& pt = points[j];
            cv::circle(canvas, pt, 5, color, -1); // 实心圆标记
            cv::putText(canvas, std::to_string(j), cv::Point(pt.x + 10, pt.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

    // 显示并保存
    cv::imwrite("C:/Users/Lee/Desktop/mic_corner_points.png",  canvas);

}





void Micromanipulator::on_Btn_calibration_MacRobot_clicked()
{
    ui->BtnGetCurrentState->click();//先获取机械臂当前位姿
    Delay(1000);
    // int RobotX = ui->XPosition->text().toInt();
    // int RobotY = ui->YPosition->text().toInt();
    // int RobotZ = ui->ZPosition->text().toInt();
    // int RobotRX = ui->RXPosition->text().toInt();
    // int RobotRY = ui->RYPosition->text().toInt();
    // int RobotRZ = ui->RZPosition->text().toInt();
}


void Micromanipulator::on_Btn_EyeHandCalibration_clicked()
{

}


//获取相机内参
void Micromanipulator::on_Btn_CameraCalibration_clicked()
{
    // === 用户自定义参数 ===
    std::string folderPath = "C:/Users/Lee/Desktop/BiaoDing/";   // 图像所在的文件夹（可改）
    cv::Size boardSize(19, 17);           // 棋盘格内角点数量（列 x 行）→ 20x18方格对应19x17内角点
    float squareSize = 0.3f;              // 单个方格边长（单位：mm 或 cm）
    // === 存储变量 ===
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;
    std::vector<cv::Point3f> objPtsTemplate;
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            objPtsTemplate.emplace_back(j * squareSize, i * squareSize, 0.0f);

    // === 遍历图像文件夹 ===
    std::vector<std::string> imageFiles;
    for (const auto& entry : std::filesystem::directory_iterator(folderPath))
    {
        imageFiles.push_back(entry.path().string());
    }

    if (imageFiles.empty())
    {
        qDebug() << "未找到任何图像文件！";
    }

    for (const auto& imagePath : imageFiles)
    {

        cv::Mat img = cv::imread(imagePath);
        if (img.empty())
        {
            qDebug() << "无法读取图像：" << imagePath;
            continue;
        }

        if (imageSize == cv::Size())
            imageSize = img.size();

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, boardSize, corners);

        if (found)
        {
            //确保角点从上往下，从左往右排列
            if(corners[0].x > corners.back().x)
            {
                //交换列
                for(int i=0; i < (int)boardSize.height; i++)  //行
                    for(int j=0; j < (int)boardSize.width/2; j++) //列
                        std::swap(corners[i*boardSize.width + j], corners[(i + 1)*boardSize.width - j - 1]);
            }
            if(corners[0].y > corners.back().y)
            {
                //交换行
                for(int i=0; i < (int)boardSize.width; i++) //列
                    for(int j=0; j < (int)boardSize.height/2; j++) //行
                        std::swap(corners[j*boardSize.width + i], corners[(boardSize.height - j - 1)*boardSize.width + i]);
            }

            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

            imagePoints.push_back(corners);
            objectPoints.push_back(objPtsTemplate);

            cv::drawChessboardCorners(img, boardSize, corners, found);
            // 特别标出第一个点（OpenCV排序起点），绿色大圆圈
            cv::circle(img, corners[0], 8, cv::Scalar(0, 255, 0), 2);
            cv::imshow(imagePath, img);
            // cv::waitKey();
        }
        else
        {
            qDebug() << "跳过未检测角点的图像: " << imagePath;
        }
    }

    // cv::destroyAllWindows();

    if (imagePoints.size() < 3)
    {
        qDebug() << "有效图像太少，至少需要3张带角点图像！";

    }


    std::vector<cv::Mat> rvecs, tvecs;

    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                                     cameraMatrix, distCoeffs, rvecs, tvecs);

    qDebug() << "\n=== 标定完成 ===\n";
    qDebug() << "重投影误差 RMS = " << rms ;
    qDebug() << "相机内参矩阵:\n"
             <<"["<<cameraMatrix.at<double>(0, 0)<<cameraMatrix.at<double>(0,1)<<cameraMatrix.at<double>(0,2)<<"]\n"
             <<"["<<cameraMatrix.at<double>(1, 0)<<cameraMatrix.at<double>(1,1)<<cameraMatrix.at<double>(1,2)<<"]\n"
             <<"["<<cameraMatrix.at<double>(2, 0)<<cameraMatrix.at<double>(2,1)<<cameraMatrix.at<double>(2,2)<<"]\n";
    qDebug()<<"畸变系数:\n" << distCoeffs.at<double>(0,0)<< distCoeffs.at<double>(0,1)<< distCoeffs.at<double>(0,2)<< distCoeffs.at<double>(0,3)<< distCoeffs.at<double>(0,4);

    // === 保存重投影误差到 CSV ===
    std::ofstream outFile("C:/Users/Lee/Desktop/projection_errors.csv");
    if (!outFile.is_open())
    {
        qDebug() << "无法打开 projection_errors.csv 文件";
        return;
    }
    outFile << "Image,PointIndex,Error(px),ProjectedX,ProjectedY,DetectedX,DetectedY\n";
    outFile << std::fixed << std::setprecision(6);  // 保留小数点后6位

    // === 计算每张图像的重投影误差 ===
    double totalError = 0;
    int totalPoints = 0;

    // 写表头（只需一次）
    outFile << "Image";
    for (size_t j = 0; j < imagePoints[0].size(); ++j)
        outFile << ",Error_" << j;
    outFile << "\n";

    for (size_t i = 0; i < imagePoints.size(); ++i)
    {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);

        double err = 0.0;
        double maxErr = 0.0;

        std::string imageName = std::filesystem::path(imageFiles[i]).filename().string();

        outFile << imageName;  // 一行开始

        for (size_t j = 0; j < imagePoints[i].size(); ++j)
        {
            double e = cv::norm(imagePoints[i][j] - projectedPoints[j]);
            err += e * e;
            if (e > maxErr)
                maxErr = e;

            outFile << "," << e;  // 横向写
        }

        outFile << "\n";  // 换行

        double meanErr = std::sqrt(err / imagePoints[i].size());
        totalError += err;
        totalPoints += (int)imagePoints[i].size();

        qDebug() << "图像" << QString::fromStdString(imageName)
                 << "平均重投影误差 = " << meanErr
                 << "，最大单点误差 = " << maxErr;
    }
    outFile.close();
    qDebug() << "重投影误差已写入 projection_errors.csv 文件";


}


void Micromanipulator::on_Btn_EyeHandCalibration_solute_clicked()
{

}



void Micromanipulator::on_Btn_KeepCenter_clicked()
{

    // 基准像素点（中心）
    cv::Mat Visual0 = (cv::Mat_<double>(3,1) << 424, 240, 1);
    cv::Mat Arm0 = H1 * Visual0;
    cv::Point2f arm0(Arm0.at<double>(0,0)/Arm0.at<double>(2,0),
                     Arm0.at<double>(1,0)/Arm0.at<double>(2,0));

    // 当前像素点
    cv::Mat Visual = (cv::Mat_<double>(3,1) << VisualPosition_X, VisualPosition_Y, 1);
    cv::Mat Arm = H1 * Visual;
    cv::Point2f arm(Arm.at<double>(0,0)/Arm.at<double>(2,0),
                    Arm.at<double>(1,0)/Arm.at<double>(2,0));

    // 得到偏移 ΔX, ΔY
    cv::Point2f deltaArm = arm0 - arm;

    // 最终目标位置（在机械臂坐标系下）
    qint32 x_new = std::round(deltaArm.x)+mic_X;
    qint32 y_new = std::round(deltaArm.y)+mic_Y;


    // 计算距离的平方（避免平方根运算，提高效率）
    double distanceSquared = std::pow(VisualPosition_X - 424, 2) +
                             std::pow(VisualPosition_Y - 240, 2);

    // 比较距离的平方与5的平方（25）
    if(distanceSquared > 25.0) {
        if (m_microArmModule.moveToPose(x_new, y_new, 0))
        {
            mic_X = x_new;
            mic_Y = y_new;
            mic_Z = 0;
        }
    }
    qDebug() << "dert_x= "
             << Visual.at<double>(0,0) << ", "
             << "dert_y= "
             << Visual.at<double>(1,0);



    qDebug()<<"目标x"<<x_new<<"目标y"<<y_new;


}

bool Micromanipulator::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == ui->CameraShow && event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        QPoint pos = mouseEvent->pos();   // 点击点在 QLabel 内部的坐标

        QPixmap pm = ui->CameraShow->pixmap();
        if (!pm.isNull()) {
            QSize imgSize = pm.size();                  // 图像实际像素大小
            QSize lblSize = ui->CameraShow->size();     // QLabel 的显示大小

            double scaleX = double(imgSize.width()) / lblSize.width();
            double scaleY = double(imgSize.height()) / lblSize.height();

            clicked_imgX = int(pos.x() * scaleX);
            clicked_imgY = int(pos.y() * scaleY);

            qDebug() << "点击像素坐标:" << clicked_imgX << clicked_imgY;
            ui->cameraTipPosition->setText(
                "(" + QString::number(clicked_imgX) + "," + QString::number(clicked_imgY) + ")"
                );
        }


        if (!H1.empty()) {
            // === ① 先用上一轮数据更新 H1_linear ===

            // ADD: 若 H1_linear 还没初始化（全 0），从 H1 拆 2x2 过来
            if (H1_linear(0,0)==0 && H1_linear(0,1)==0 &&
                H1_linear(1,0)==0 && H1_linear(1,1)==0) {
                H1.convertTo(H1, CV_64F);
                H1_linear = cv::Matx22d(H1.at<double>(0,0), H1.at<double>(0,1),
                                        H1.at<double>(1,0), H1.at<double>(1,1));
            }

            // 在线修正（可选算法）
            if (has_last_sample) {
                cv::Vec2d delta_pixel(
                    VisualPosition_X - last_visual.x,
                    VisualPosition_Y - last_visual.y
                    );
                if (cv::norm(delta_pixel) > 0.5) { // 小于0.5像素就忽略
                    online_samples.emplace_back(delta_pixel, last_delta_arm);
                    if ((int)online_samples.size() > swls_window_size) {
                        online_samples.pop_front();
                    }

                    switch (online_update_method) {
                    case OnlineUpdateMethod::Broyden:
                        H1_linear = updateMappingBroyden(
                            H1_linear, delta_pixel, last_delta_arm,
                            lambda_broyden, gamma_broyden
                            );
                        break;
                    case OnlineUpdateMethod::SlidingWindowLeastSquares:
                        H1_linear = updateMappingSwls(
                            online_samples, H1_linear, swls_ridge
                            );
                        break;
                    case OnlineUpdateMethod::RecursiveLeastSquaresFF:
                        H1_linear = updateMappingRlsFf(
                            delta_pixel, last_delta_arm,
                            rls_forgetting_factor, rls_initial_cov,
                            H1_linear, rls_theta, rls_cov, rls_initialized
                            );
                        break;
                    }
                }
                has_last_sample = false;
            }



            // === ② 当前点击误差 → 机械臂差值 ===
            cv::Vec2d delta_pixel_target(
                clicked_imgX - VisualPosition_X,
                clicked_imgY - VisualPosition_Y
                );
            cv::Vec2d delta_arm_cmd = H1_linear * delta_pixel_target;

            int dert_x = std::round(delta_arm_cmd[0]);
            int dert_y = std::round(delta_arm_cmd[1]);

            qDebug() << "映射机械臂坐标差值:" << dert_x << dert_y;
            if (m_microArmModule.moveToPose(dert_x + mic_X, dert_y + mic_Y, 0))
            {
                mic_X += dert_x;
                mic_Y += dert_y;
                mic_Z = 0;
            }

            ui->robotTipPosition->setText(
                "映射机械臂：(" + QString::number(dert_x + mic_X) + "," +
                QString::number(dert_y + mic_Y) + ")"
                );

            // === ③ 保存本次参考，供下次更新使用 ===
            last_visual = cv::Point2d(VisualPosition_X, VisualPosition_Y);
            last_delta_arm = delta_arm_cmd;
            has_last_sample = true;
        }



        return true; // 事件已处理
    }
    return QMainWindow::eventFilter(obj, event); // 交给父类处理其他事件
}

void Micromanipulator::on_BtnMoveVia_clicked()
{
    ui->BtnGetCurrentState->click();
    Delay(500);

    int currentAngle = ui->RZPosition->text().toInt();   // 当前角度，弧度*1000
    int targetAngle  = ui->RZPosition_2->text().toInt(); // 目标角度，弧度*1000
    cv::Mat updatedHomography;
    cv::Matx22d updatedLinear;
    QString errorMessage;

    if (!m_macroMicroController.executeYawArcMove(currentAngle,
                                                  targetAngle,
                                                  alpha_L,
                                                  3142,
                                                  0,
                                                  H_calib,
                                                  alpha_calib_rad,
                                                  1080 / 2.0,
                                                  720 / 2.0,
                                                  updatedHomography,
                                                  updatedLinear,
                                                  &errorMessage)) {
        if (!errorMessage.isEmpty()) {
            QMessageBox::warning(this, tr("警告"), errorMessage, QMessageBox::Ok);
        }
        return;
    }

    H1 = updatedHomography;
    H1_linear = updatedLinear;  // 同步 2×2
    resetOnlineUpdateState();

    qDebug()<<"dert_alpha = "<<(targetAngle / 1000.0 - alpha_calib_rad);
    qDebug()<<H1.at<double>(0,0)<<H1.at<double>(0,1)<<H1.at<double>(0,2)
             <<H1.at<double>(1,0)<<H1.at<double>(1,1)<<H1.at<double>(1,2)
             <<H1.at<double>(2,0)<<H1.at<double>(2,1)<<H1.at<double>(2,2);

}

void Micromanipulator::resetOnlineUpdateState()
{
    online_samples.clear();
    has_last_sample = false;
    rls_initialized = false;
    rls_theta = flattenLinearMapping(H1_linear);
    rls_cov = cv::Matx44d::eye() * rls_initial_cov;
}

cv::Point2f Micromanipulator::adaptiveSmoothTip(const cv::Point2f &measuredTip, bool hasMeasurement)
{
    if (!hasMeasurement) {
        return m_hasFilteredTipPixel ? m_filteredTipPixel : cv::Point2f();
    }

    if (!m_hasFilteredTipPixel) {
        m_filteredTipPixel = measuredTip;
        m_hasFilteredTipPixel = true;
        m_lastMeasuredTip = measuredTip;
        m_hasLastMeasuredTip = true;
        return m_filteredTipPixel;
    }

    float moveDistance = 0.0f;
    if (m_hasLastMeasuredTip) {
        const cv::Point2f diff = measuredTip - m_lastMeasuredTip;
        moveDistance = std::sqrt(diff.x * diff.x + diff.y * diff.y);
    }
    m_lastMeasuredTip = measuredTip;
    m_hasLastMeasuredTip = true;

    double alpha = m_tipSmoothingAlphaMoving;
    if (moveDistance < static_cast<float>(m_tipMotionThresholdPixels)) {
        alpha = m_tipSmoothingAlphaStatic;
    }

    const float alphaF = static_cast<float>(alpha);
    m_filteredTipPixel = alphaF * measuredTip + (1.0f - alphaF) * m_filteredTipPixel;
    return m_filteredTipPixel;
}

cv::Point2f Micromanipulator::kalmanPredictiveSmooth(const cv::Point2f &measuredTip, bool hasMeasurement)
{
    if (!m_kalmanInitialized) {
        if (!hasMeasurement) {
            return m_hasFilteredTipPixel ? m_filteredTipPixel : cv::Point2f();
        }

        const float dt = 1.0f;
        m_tipKalmanFilter.init(4, 2, 0, CV_32F);
        m_tipKalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4)
                                              << 1, 0, dt, 0,
                                                 0, 1, 0, dt,
                                                 0, 0, 1, 0,
                                                 0, 0, 0, 1);
        m_tipKalmanFilter.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
        m_tipKalmanFilter.measurementMatrix.at<float>(0, 0) = 1.0f;
        m_tipKalmanFilter.measurementMatrix.at<float>(1, 1) = 1.0f;
        m_tipKalmanFilter.processNoiseCov = (cv::Mat_<float>(4, 4)
                                             << m_kalmanProcessNoisePos, 0, 0, 0,
                                                0, m_kalmanProcessNoisePos, 0, 0,
                                                0, 0, m_kalmanProcessNoiseVel, 0,
                                                0, 0, 0, m_kalmanProcessNoiseVel);
        m_tipKalmanFilter.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * m_kalmanMeasurementNoise;
        m_tipKalmanFilter.errorCovPost = cv::Mat::eye(4, 4, CV_32F) * m_kalmanInitialError;
        m_tipKalmanFilter.statePost = (cv::Mat_<float>(4, 1)
                                       << measuredTip.x, measuredTip.y, 0.0f, 0.0f);
        m_kalmanInitialized = true;
        m_filteredTipPixel = measuredTip;
        m_hasFilteredTipPixel = true;
        return m_filteredTipPixel;
    }

    const float dt = 1.0f;
    m_tipKalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4)
                                          << 1, 0, dt, 0,
                                             0, 1, 0, dt,
                                             0, 0, 1, 0,
                                             0, 0, 0, 1);

    m_tipKalmanFilter.processNoiseCov = (cv::Mat_<float>(4, 4)
                                         << m_kalmanProcessNoisePos, 0, 0, 0,
                                            0, m_kalmanProcessNoisePos, 0, 0,
                                            0, 0, m_kalmanProcessNoiseVel, 0,
                                            0, 0, 0, m_kalmanProcessNoiseVel);
    m_tipKalmanFilter.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * m_kalmanMeasurementNoise;

    cv::Mat prediction = m_tipKalmanFilter.predict();
    cv::Mat estimate = prediction;

    if (hasMeasurement) {
        const cv::Mat measurement = (cv::Mat_<float>(2, 1) << measuredTip.x, measuredTip.y);
        estimate = m_tipKalmanFilter.correct(measurement);
    }

    m_filteredTipPixel = cv::Point2f(static_cast<float>(estimate.at<float>(0)),
                                     static_cast<float>(estimate.at<float>(1)));
    m_hasFilteredTipPixel = true;
    return m_filteredTipPixel;
}

cv::Point2f Micromanipulator::smoothTip(const cv::Point2f &measuredTip, bool hasMeasurement)
{
    if (m_tipSmoothingMode == TipSmoothingMode::KalmanPredictive) {
        return kalmanPredictiveSmooth(measuredTip, hasMeasurement);
    }

    if (m_tipSmoothingMode == TipSmoothingMode::GlobalFixed) {
        if (!hasMeasurement) {
            return m_hasFilteredTipPixel ? m_filteredTipPixel : cv::Point2f();
        }

        if (!m_hasFilteredTipPixel) {
            m_filteredTipPixel = measuredTip;
            m_hasFilteredTipPixel = true;
            return m_filteredTipPixel;
        }

        const float alphaF = static_cast<float>(m_tipSmoothingAlpha);
        m_filteredTipPixel = alphaF * measuredTip + (1.0f - alphaF) * m_filteredTipPixel;
        return m_filteredTipPixel;
    }

    return adaptiveSmoothTip(measuredTip, hasMeasurement);
}

cv::Rect Micromanipulator::buildFocusRoi(const cv::Mat &bgrFrame, cv::Mat &roiOut,
                                         cv::Point2f &roiCenter, bool &roiAnchoredToTip,
                                         cv::Point2f &rawRoiCenter)
{
    roiAnchoredToTip = false;
    roiCenter = cv::Point2f(static_cast<float>(bgrFrame.cols) / 2.0f,
                            static_cast<float>(bgrFrame.rows) / 2.0f);
    rawRoiCenter = roiCenter;

    if (bgrFrame.empty()) {
        roiOut.release();
        m_hasLastFocusRoi = false;
        m_hasFilteredFocusCenter = false;
        return cv::Rect();
    }

    ImageProcessor::FrameRequest request;
    request.frame = bgrFrame;
    request.applyCalibration = ui->cBtnCameraCalibrated->isChecked();
    if (request.applyCalibration) {
        request.cameraMatrix = cameraMatrix;
        request.distCoeffs = distCoeffs;
    }
    request.flipImage = ui->rBtnImageFlip->isChecked();
    if (!BackImage.empty()) {
        request.background = BackImage;
    }
    request.displayMode = ImageProcessor::DisplayMode::Original;

    const ImageProcessor::ProcessedImage processed = runImageProcessingPipeline(request);

    if (processed.hasTip) {
        roiCenter = processed.filteredTip;
        roiAnchoredToTip = true;
        m_lastFocusTip = processed.filteredTip;
        m_hasLastFocusTip = true;
    } else if (m_hasFilteredTipPixel) {
        roiCenter = m_filteredTipPixel;
        roiAnchoredToTip = true;
        m_lastFocusTip = m_filteredTipPixel;
        m_hasLastFocusTip = true;
    } else if (m_hasLastFocusTip) {
        roiCenter = m_lastFocusTip;
    } else if (m_hasLastFocusRoi) {
        roiCenter = cv::Point2f(static_cast<float>(m_lastFocusRoi.x + m_lastFocusRoi.width / 2.0),
                                static_cast<float>(m_lastFocusRoi.y + m_lastFocusRoi.height / 2.0));
    }

    rawRoiCenter = roiCenter;

    if (!m_hasFilteredFocusCenter) {
        m_filteredFocusCenter = roiCenter;
        m_hasFilteredFocusCenter = true;
    } else {
        const double alpha = std::clamp(m_focusRoiSmoothingAlpha, 0.0, 1.0);
        cv::Point2f blended(static_cast<float>(alpha) * roiCenter +
                            static_cast<float>(1.0 - alpha) * m_filteredFocusCenter);

        cv::Point2f delta = blended - m_filteredFocusCenter;
        const double deltaNorm = std::sqrt(delta.x * delta.x + delta.y * delta.y);
        if (deltaNorm > m_focusRoiMaxJumpPixels && deltaNorm > 1e-3) {
            const float scale = static_cast<float>(m_focusRoiMaxJumpPixels / deltaNorm);
            delta *= scale;
            blended = m_filteredFocusCenter + delta;
        }

        m_filteredFocusCenter = blended;
    }

    roiCenter = m_filteredFocusCenter;
    m_lastRawFocusCenter = rawRoiCenter;
    m_lastFocusRoiAnchoredToTip = roiAnchoredToTip;

    const int requestedSize = std::max(32, ui->spinFocusRoiSize->value());
    const int half = requestedSize / 2;
    const int cx = static_cast<int>(std::round(roiCenter.x));
    const int cy = static_cast<int>(std::round(roiCenter.y));

    int x = 0;
    int y = 0;
    if (roiAnchoredToTip) {
        // 让针尖落在 ROI 的右边框，便于保持针尖在 ROI 的视场边缘。
        x = cx - requestedSize;
        y = cy - half;
    } else {
        x = cx - half;
        y = cy - half;
    }
    x = std::clamp(x, 0, std::max(0, bgrFrame.cols - 1));
    y = std::clamp(y, 0, std::max(0, bgrFrame.rows - 1));

    int width = std::min(requestedSize, bgrFrame.cols - x);
    int height = std::min(requestedSize, bgrFrame.rows - y);

    // 若 ROI 因靠近边界缩小，保证至少留下 1x1 的有效区域。
    width = std::max(1, width);
    height = std::max(1, height);

    const cv::Rect roiRect(x, y, width, height);
    roiOut = bgrFrame(roiRect).clone();

    const cv::Point2f realizedCenter(static_cast<float>(roiRect.x + roiRect.width / 2.0),
                                     static_cast<float>(roiRect.y + roiRect.height / 2.0));
    m_filteredFocusCenter = realizedCenter;
    roiCenter = realizedCenter;

    m_lastFocusRoi = roiRect;
    m_hasLastFocusRoi = true;
    return roiRect;
}

QString Micromanipulator::focusRoiSourceToString(bool roiAnchoredToTip) const
{
    return roiAnchoredToTip ? QStringLiteral("tip_aligned")
                            : QStringLiteral("fallback");
}

cv::Mat Micromanipulator::normalizeTo8U(const cv::Mat &gray) const
{
    if (gray.type() == CV_8U) {
        return gray;
    }

    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);

    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(grayFloat, &minVal, &maxVal);
    if (maxVal - minVal < 1e-6) {
        return cv::Mat(gray.size(), CV_8U, cv::Scalar(0));
    }

    cv::Mat normalized;
    grayFloat.convertTo(normalized, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    return normalized;
}

namespace
{
double normalizePositive(double value, double scale)
{
    if (!std::isfinite(value) || value <= 0.0) {
        return 0.0;
    }
    return std::clamp(value / (value + scale), 0.0, 1.0);
}
}

Micromanipulator::FocusMeasureSample Micromanipulator::evaluateFocusMeasures(const cv::Mat &bgrFrame, int stepIndex, int targetZ) const
{
    FocusMeasureSample sample;
    sample.stepIndex = stepIndex;
    sample.targetZ = targetZ;

    cv::Mat gray;
    if (bgrFrame.channels() == 3) {
        cv::cvtColor(bgrFrame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = bgrFrame.clone();
    }

    gray = normalizeTo8U(gray);

    sample.variance = focusVariance(gray);
    sample.tenengrad = focusTenengrad(gray);
    sample.brenner = focusBrenner(gray);
    sample.laplacianVar = focusLaplacianVariance(gray);
    sample.entropy = focusEntropy(gray);
    sample.highFrequency = focusHighFrequencyEnergy(gray);
    return sample;
}

double Micromanipulator::focusVariance(const cv::Mat &gray) const
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    constexpr double kMaxVariance = 255.0 * 255.0; // 8-bit 输入的理论最大方差
    const double variance = stddev[0] * stddev[0];
    return std::clamp(variance / kMaxVariance, 0.0, 1.0);
}

double Micromanipulator::focusTenengrad(const cv::Mat &gray) const
{
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_64F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_64F, 0, 1, 3);
    cv::Mat mag2 = gx.mul(gx) + gy.mul(gy);
    const double avgMag2 = cv::sum(mag2)[0] / static_cast<double>(gray.total());
    constexpr double kExpectedScale = 1.0e5; // 使梯度能量在 0~1 内平滑分布的经验尺度
    return normalizePositive(avgMag2, kExpectedScale);
}

double Micromanipulator::focusBrenner(const cv::Mat &gray) const
{
    cv::Mat gray32f;
    gray.convertTo(gray32f, CV_32F);
    double value = 0.0;

    for (int y = 0; y < gray32f.rows; ++y) {
        const float *row = gray32f.ptr<float>(y);
        for (int x = 0; x < gray32f.cols - 2; ++x) {
            const double diff = static_cast<double>(row[x] - row[x + 2]);
            value += diff * diff;
        }
    }
    const double avgValue = value / static_cast<double>(gray.total());
    constexpr double kMaxDiffSquare = 255.0 * 255.0;
    return std::clamp(avgValue / kMaxDiffSquare, 0.0, 1.0);
}

double Micromanipulator::focusLaplacianVariance(const cv::Mat &gray) const
{
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    const double lapVar = stddev[0] * stddev[0];
    constexpr double kExpectedScale = 1.0e5;
    return normalizePositive(lapVar, kExpectedScale);
}

double Micromanipulator::focusEntropy(const cv::Mat &gray) const
{
    CV_Assert(gray.type() == CV_8U);
    int histSize = 256;
    float range[] = {0, 256};
    const float *ranges[] = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, ranges, true, false);
    hist /= static_cast<double>(gray.total());

    double entropy = 0.0;
    for (int i = 0; i < histSize; ++i) {
        const double p = hist.at<float>(i);
        if (p > 1e-12) {
            entropy -= p * std::log2(p);
        }
    }
    constexpr double kMaxEntropy = 8.0; // 8-bit 灰度的最大熵
    return std::clamp(entropy / kMaxEntropy, 0.0, 1.0);
}

double Micromanipulator::focusHighFrequencyEnergy(const cv::Mat &gray) const
{
    CV_Assert(gray.type() == CV_8U);
    static const cv::Mat highPassKernel = (cv::Mat_<double>(3, 3) <<
                                            -1, -1, -1,
                                            -1,  8, -1,
                                            -1, -1, -1);

    cv::Mat filtered;
    cv::filter2D(gray, filtered, CV_64F, highPassKernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::Mat absFiltered = cv::abs(filtered);
    const double meanEnergy = cv::sum(absFiltered)[0] / static_cast<double>(gray.total());

    constexpr double kExpectedScale = 2040.0; // 近似 8 * 255，保证数值压缩到 0~1
    return normalizePositive(meanEnergy, kExpectedScale);
}

QString Micromanipulator::buildDefaultFocusLogPath() const
{
    QString baseDir = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation);
    if (baseDir.isEmpty()) {
        baseDir = QStringLiteral("C:/Users/Lee/Desktop");
    }
    QDir dir(baseDir.isEmpty() ? QDir::homePath() : baseDir);
    if (!dir.exists()) {
        dir.mkpath(".");
    }
    return dir.filePath(QStringLiteral("autofocus_%1.csv")
                            .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")));
}

bool Micromanipulator::startFocusLogRecording(const QString &filePath)
{
    stopFocusLogRecording();

    QString outputPath = filePath.trimmed();
    if (outputPath.isEmpty()) {
        outputPath = buildDefaultFocusLogPath();
    }

    if (!outputPath.endsWith(".csv", Qt::CaseInsensitive)) {
        const QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        QDir outputDir(outputPath);
        if (!outputDir.exists()) {
            outputDir.mkpath(".");
        }
        outputPath = outputDir.filePath(QStringLiteral("autofocus_%1.csv").arg(timestamp));
    }

    QFileInfo fileInfo(outputPath);
    if (!fileInfo.dir().exists()) {
        QDir().mkpath(fileInfo.dir().path());
    }

    m_focusLogFilePath = outputPath;
    m_focusLogFile.setFileName(m_focusLogFilePath);
    if (!m_focusLogFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("提示"), tr("无法创建自动聚焦日志：%1").arg(m_focusLogFilePath));
        m_focusLogFilePath.clear();
        return false;
    }

    m_focusLogStream.setDevice(&m_focusLogFile);
    m_focusLogStream.setRealNumberPrecision(6);
    m_focusLogIndex = 0;
    m_isFocusLogRecording = true;
    ui->lineFocusOutputPath->setText(m_focusLogFilePath);

    m_focusLogStream << "index,phase,z,roi_x,roi_y,roi_w,roi_h,roi_center_raw_x,roi_center_raw_y,roi_center_x,roi_center_y,roi_source,"
                        "tenengrad\n";
    m_focusLogStream.flush();
    return true;
}

void Micromanipulator::stopFocusLogRecording()
{
    if (m_focusLogFile.isOpen()) {
        m_focusLogFile.close();
    }
    m_isFocusLogRecording = false;
}

QString Micromanipulator::buildDefaultTipErrorLogPath() const
{
    const QString baseDir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QDir dir(baseDir.isEmpty() ? QDir::homePath() : baseDir);
    dir.mkpath("tip_error_logs");
    return dir.filePath(QStringLiteral("tip_error_logs/tip_error_%1.csv")
                            .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")));
}

void Micromanipulator::startTipErrorRecording(const QString &filePath)
{
    stopTipErrorRecording();

    m_tipErrorFilePath = filePath;
    m_tipErrorFile.setFileName(m_tipErrorFilePath);
    if (!m_tipErrorFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("提示"), tr("无法创建误差记录文件：%1").arg(m_tipErrorFilePath));
        return;
    }

    m_tipErrorStream.setDevice(&m_tipErrorFile);
    m_tipErrorStream.setRealNumberPrecision(6);
    m_tipErrorFrameIndex = 0;
    m_isTipErrorRecording = true;

    m_logAdaptiveFilteredTip = cv::Point2f(0.0f, 0.0f);
    m_logAdaptiveHasFilteredTip = false;
    m_logAdaptiveLastMeasuredTip = cv::Point2f(0.0f, 0.0f);
    m_logAdaptiveHasLastMeasuredTip = false;
    m_logGlobalFilteredTip = cv::Point2f(0.0f, 0.0f);
    m_logGlobalHasFilteredTip = false;
    m_logKalmanFilter = cv::KalmanFilter();
    m_logKalmanInitialized = false;
    m_logKalmanFilteredTip = cv::Point2f(0.0f, 0.0f);
    m_logKalmanHasFilteredTip = false;

    m_tipErrorStream << "frame_index,timestamp,measured_x,measured_y,"
                        "adaptive_x,adaptive_y,adaptive_error_x,adaptive_error_y,"
                        "global_x,global_y,global_error_x,global_error_y,"
                        "kalman_x,kalman_y,kalman_error_x,kalman_error_y\n";
}

void Micromanipulator::stopTipErrorRecording()
{
    if (!m_isTipErrorRecording) {
        return;
    }

    m_tipErrorStream.flush();
    m_tipErrorFile.close();
    m_isTipErrorRecording = false;
}

void Micromanipulator::writeTipErrorSample(const ImageProcessor::ProcessedImage &result)
{
    if (!m_tipErrorFile.isOpen()) {
        return;
    }

    const QString timestamp = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);

    double measuredX = std::numeric_limits<double>::quiet_NaN();
    double measuredY = std::numeric_limits<double>::quiet_NaN();
    double adaptiveX = std::numeric_limits<double>::quiet_NaN();
    double adaptiveY = std::numeric_limits<double>::quiet_NaN();
    double adaptiveErrX = std::numeric_limits<double>::quiet_NaN();
    double adaptiveErrY = std::numeric_limits<double>::quiet_NaN();
    double globalX = std::numeric_limits<double>::quiet_NaN();
    double globalY = std::numeric_limits<double>::quiet_NaN();
    double globalErrX = std::numeric_limits<double>::quiet_NaN();
    double globalErrY = std::numeric_limits<double>::quiet_NaN();
    double kalmanX = std::numeric_limits<double>::quiet_NaN();
    double kalmanY = std::numeric_limits<double>::quiet_NaN();
    double kalmanErrX = std::numeric_limits<double>::quiet_NaN();
    double kalmanErrY = std::numeric_limits<double>::quiet_NaN();

    if (result.hasMeasuredTip) {
        measuredX = result.measuredTip.x;
        measuredY = result.measuredTip.y;

        const cv::Point2f measuredTip(static_cast<float>(measuredX),
                                      static_cast<float>(measuredY));

        auto adaptiveSmooth = [this](const cv::Point2f &tip, bool hasMeasurement) {
            if (!hasMeasurement) {
                return m_logAdaptiveHasFilteredTip ? m_logAdaptiveFilteredTip : cv::Point2f();
            }

            if (!m_logAdaptiveHasFilteredTip) {
                m_logAdaptiveFilteredTip = tip;
                m_logAdaptiveHasFilteredTip = true;
                m_logAdaptiveLastMeasuredTip = tip;
                m_logAdaptiveHasLastMeasuredTip = true;
                return m_logAdaptiveFilteredTip;
            }

            float moveDistance = 0.0f;
            if (m_logAdaptiveHasLastMeasuredTip) {
                const cv::Point2f diff = tip - m_logAdaptiveLastMeasuredTip;
                moveDistance = std::sqrt(diff.x * diff.x + diff.y * diff.y);
            }
            m_logAdaptiveLastMeasuredTip = tip;
            m_logAdaptiveHasLastMeasuredTip = true;

            double alpha = m_tipSmoothingAlphaMoving;
            if (moveDistance < static_cast<float>(m_tipMotionThresholdPixels)) {
                alpha = m_tipSmoothingAlphaStatic;
            }

            const float alphaF = static_cast<float>(alpha);
            m_logAdaptiveFilteredTip = alphaF * tip + (1.0f - alphaF) * m_logAdaptiveFilteredTip;
            return m_logAdaptiveFilteredTip;
        };

        auto globalSmooth = [this](const cv::Point2f &tip, bool hasMeasurement) {
            if (!hasMeasurement) {
                return m_logGlobalHasFilteredTip ? m_logGlobalFilteredTip : cv::Point2f();
            }

            if (!m_logGlobalHasFilteredTip) {
                m_logGlobalFilteredTip = tip;
                m_logGlobalHasFilteredTip = true;
                return m_logGlobalFilteredTip;
            }

            const float alphaF = static_cast<float>(m_tipSmoothingAlpha);
            m_logGlobalFilteredTip = alphaF * tip + (1.0f - alphaF) * m_logGlobalFilteredTip;
            return m_logGlobalFilteredTip;
        };

        auto kalmanSmooth = [this](const cv::Point2f &tip, bool hasMeasurement) {
            if (!m_logKalmanInitialized) {
                if (!hasMeasurement) {
                    return m_logKalmanHasFilteredTip ? m_logKalmanFilteredTip : cv::Point2f();
                }

                const float dt = 1.0f;
                m_logKalmanFilter.init(4, 2, 0, CV_32F);
                m_logKalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4)
                                                      << 1, 0, dt, 0,
                                                         0, 1, 0, dt,
                                                         0, 0, 1, 0,
                                                         0, 0, 0, 1);
                m_logKalmanFilter.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
                m_logKalmanFilter.measurementMatrix.at<float>(0, 0) = 1.0f;
                m_logKalmanFilter.measurementMatrix.at<float>(1, 1) = 1.0f;
                m_logKalmanFilter.processNoiseCov = (cv::Mat_<float>(4, 4)
                                                     << m_kalmanProcessNoisePos, 0, 0, 0,
                                                        0, m_kalmanProcessNoisePos, 0, 0,
                                                        0, 0, m_kalmanProcessNoiseVel, 0,
                                                        0, 0, 0, m_kalmanProcessNoiseVel);
                m_logKalmanFilter.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * m_kalmanMeasurementNoise;
                m_logKalmanFilter.errorCovPost = cv::Mat::eye(4, 4, CV_32F) * m_kalmanInitialError;
                m_logKalmanFilter.statePost = (cv::Mat_<float>(4, 1)
                                               << tip.x, tip.y, 0.0f, 0.0f);
                m_logKalmanInitialized = true;
                m_logKalmanFilteredTip = tip;
                m_logKalmanHasFilteredTip = true;
                return m_logKalmanFilteredTip;
            }

            const float dt = 1.0f;
            m_logKalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4)
                                                  << 1, 0, dt, 0,
                                                     0, 1, 0, dt,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1);
            m_logKalmanFilter.processNoiseCov = (cv::Mat_<float>(4, 4)
                                                 << m_kalmanProcessNoisePos, 0, 0, 0,
                                                    0, m_kalmanProcessNoisePos, 0, 0,
                                                    0, 0, m_kalmanProcessNoiseVel, 0,
                                                    0, 0, 0, m_kalmanProcessNoiseVel);
            m_logKalmanFilter.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * m_kalmanMeasurementNoise;

            cv::Mat prediction = m_logKalmanFilter.predict();
            cv::Mat estimate = prediction;
            if (hasMeasurement) {
                const cv::Mat measurement = (cv::Mat_<float>(2, 1) << tip.x, tip.y);
                estimate = m_logKalmanFilter.correct(measurement);
            }

            m_logKalmanFilteredTip = cv::Point2f(static_cast<float>(estimate.at<float>(0)),
                                                 static_cast<float>(estimate.at<float>(1)));
            m_logKalmanHasFilteredTip = true;
            return m_logKalmanFilteredTip;
        };

        const cv::Point2f adaptiveTip = adaptiveSmooth(measuredTip, true);
        const cv::Point2f globalTip = globalSmooth(measuredTip, true);
        const cv::Point2f kalmanTip = kalmanSmooth(measuredTip, true);

        adaptiveX = adaptiveTip.x;
        adaptiveY = adaptiveTip.y;
        adaptiveErrX = adaptiveX - measuredX;
        adaptiveErrY = adaptiveY - measuredY;
        globalX = globalTip.x;
        globalY = globalTip.y;
        globalErrX = globalX - measuredX;
        globalErrY = globalY - measuredY;
        kalmanX = kalmanTip.x;
        kalmanY = kalmanTip.y;
        kalmanErrX = kalmanX - measuredX;
        kalmanErrY = kalmanY - measuredY;
    }

    m_tipErrorStream << m_tipErrorFrameIndex << ','
                     << timestamp << ','
                     << measuredX << ','
                     << measuredY << ','
                     << adaptiveX << ','
                     << adaptiveY << ','
                     << adaptiveErrX << ','
                     << adaptiveErrY << ','
                     << globalX << ','
                     << globalY << ','
                     << globalErrX << ','
                     << globalErrY << ','
                     << kalmanX << ','
                     << kalmanY << ','
                     << kalmanErrX << ','
                     << kalmanErrY << '\n';
    ++m_tipErrorFrameIndex;
}
