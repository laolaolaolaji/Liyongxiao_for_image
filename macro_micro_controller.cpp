#include "macro_micro_controller.h"

#include "camera_module.h"
#include "image_processor.h"
#include "macro_arm.h"
#include "micro_arm.h"

#include <QCoreApplication>
#include <QDebug>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSerialPort>

#include <cmath>

namespace
{
/** 常量：角度转弧度。 */
constexpr double kDegToRad = M_PI / 180.0;

/**
 * @brief 使用 Kåsa 最小二乘算法拟合圆心和半径。
 * @param points 平面点集合。
 * @param cx     拟合的圆心 x。
 * @param cy     拟合的圆心 y。
 * @param radius 拟合的半径。
 */
inline void fitCircleKasa(const std::vector<cv::Point2d> &points,
                          double &cx, double &cy, double &radius)
{
    const int n = static_cast<int>(points.size());
    cv::Mat1d A(n, 3), b(n, 1);
    for (int i = 0; i < n; ++i) {
        const double x = points[i].x;
        const double y = points[i].y;
        A(i, 0) = x;
        A(i, 1) = y;
        A(i, 2) = 1.0;
        b(i, 0) = -(x * x + y * y);
    }

    // 通过 SVD 求解线性方程，提升数值稳定性。
    cv::Mat1d coeffs;
    cv::solve(A, b, coeffs, cv::DECOMP_SVD);

    const double a = coeffs(0, 0);
    const double bcoef = coeffs(1, 0);
    const double c = coeffs(2, 0);

    cx = -a / 2.0;
    cy = -bcoef / 2.0;
    radius = std::sqrt(std::max(0.0, cx * cx + cy * cy - c));
}

/**
 * @brief 将拟合圆与 α 序列对齐，确定旋转方向和相位。
 */
inline void fitPhaseWithAlpha(const std::vector<cv::Point2d> &points,
                              const std::vector<double> &alphas,
                              double cx, double cy,
                              int &direction, double &phi, double &rmse)
{
    // 内部 lambda：固定方向求解 φ 与误差。
    auto solveForDirection = [&](int sign, double &phiOut, double &rmseOut) {
        double csum = 0.0;
        double ssum = 0.0;
        const int n = static_cast<int>(points.size());
        for (int i = 0; i < n; ++i) {
            const double theta = std::atan2(points[i].y - cy, points[i].x - cx);
            const double angle = theta - sign * alphas[i];
            csum += std::cos(angle);
            ssum += std::sin(angle);
        }
        phiOut = std::atan2(ssum, csum);

        // 计算相位拟合的均方误差，用于比较 ±1 两种方向。
        double squaredError = 0.0;
        for (int i = 0; i < n; ++i) {
            const double theta = std::atan2(points[i].y - cy, points[i].x - cx);
            double diff = theta - (phiOut + sign * alphas[i]);
            diff = std::atan2(std::sin(diff), std::cos(diff));
            squaredError += diff * diff;
        }
        rmseOut = std::sqrt(squaredError / n);
    };

    double phiPositive = 0.0;
    double rmsePositive = 0.0;
    solveForDirection(+1, phiPositive, rmsePositive);

    double phiNegative = 0.0;
    double rmseNegative = 0.0;
    solveForDirection(-1, phiNegative, rmseNegative);

    if (rmsePositive <= rmseNegative) {
        direction = +1;
        phi = phiPositive;
        rmse = rmsePositive;
    } else {
        direction = -1;
        phi = phiNegative;
        rmse = rmseNegative;
    }
}

/**
 * @brief 根据拟合结果构造 3x4 L 矩阵。
 */
inline cv::Mat buildL(double cx, double cy, double radius, double phi, double zIntercept, double zSlope)
{
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    return (cv::Mat_<double>(3, 4) << cx, radius * c, -radius * s, 0.0,
            cy, radius * s, radius * c, 0.0,
            zIntercept, 0.0, 0.0, zSlope);
}

/**
 * @brief 综合几何与 α 信息，计算 yaw 对应的 L 矩阵。
 */
cv::Mat fitCircle2DGeometryThenAlpha(const std::vector<cv::Vec4d> &pts,
                                     double *cx = nullptr,
                                     double *cy = nullptr,
                                     double *radius = nullptr,
                                     double *phi = nullptr,
                                     int *direction = nullptr,
                                     double *rmseAlpha = nullptr,
                                     bool print = true)
{
    // 步骤 0：校验输入点数量，至少需要三个点才能拟合圆。
    const int n = static_cast<int>(pts.size());
    if (n < 3) {
        if (print) {
            qWarning() << "需要 ≥3 点";
        }
        return cv::Mat();
    }

    // 步骤 1：拆分几何信息和角度信息，拟合 z = k * alpha + b。
    std::vector<cv::Point2d> points(n);
    std::vector<double> alphas(n);
    std::vector<double> zs(n);
    for (int i = 0; i < n; ++i) {
        points[i] = {pts[i][0], pts[i][1]};
        alphas[i] = pts[i][3];
        zs[i] = pts[i][2];
    }

    double alphaMean = 0.0;
    double zMean = 0.0;
    for (int i = 0; i < n; ++i) {
        alphaMean += alphas[i];
        zMean += zs[i];
    }
    alphaMean /= n;
    zMean /= n;

    double denom = 0.0;
    double numer = 0.0;
    for (int i = 0; i < n; ++i) {
        const double da = alphas[i] - alphaMean;
        denom += da * da;
        numer += da * (zs[i] - zMean);
    }
    const double zSlope = (std::abs(denom) > 1e-9) ? (numer / denom) : 0.0;
    const double zIntercept = zMean - zSlope * alphaMean;

    // 步骤 2：只根据 (x,y) 拟合圆心和半径。
    double cxLocal = 0.0;
    double cyLocal = 0.0;
    double rLocal = 0.0;
    fitCircleKasa(points, cxLocal, cyLocal, rLocal);

    // 步骤 3：结合 α 找到最优相位与方向，得到整体 yaw 模型。
    int directionLocal = +1;
    double phiLocal = 0.0;
    double rmseLocal = 0.0;
    fitPhaseWithAlpha(points, alphas, cxLocal, cyLocal, directionLocal, phiLocal, rmseLocal);

    // 步骤 4：根据调用者需要输出各个中间量。
    if (cx) *cx = cxLocal;
    if (cy) *cy = cyLocal;
    if (radius) *radius = rLocal;
    if (phi) *phi = phiLocal;
    if (direction) *direction = directionLocal;
    if (rmseAlpha) *rmseAlpha = rmseLocal;

    // 步骤 5：生成最终 L 矩阵。
    cv::Mat L = buildL(cxLocal, cyLocal, rLocal, phiLocal, zIntercept, zSlope);

    if (print) {
        // 步骤 6：输出调试信息，包含半径一致性与 L 矩阵内容。
        double meanR = 0.0;
        for (const auto &pt : points) {
            const double dx = pt.x - cxLocal;
            const double dy = pt.y - cyLocal;
            meanR += std::sqrt(dx * dx + dy * dy);
        }
        meanR /= n;

        double varR = 0.0;
        for (const auto &pt : points) {
            const double dx = pt.x - cxLocal;
            const double dy = pt.y - cyLocal;
            const double ri = std::sqrt(dx * dx + dy * dy);
            varR += (ri - meanR) * (ri - meanR);
        }
        varR /= std::max(1, n - 1);
        const double stdR = std::sqrt(varR);

        qInfo() << "圆心(cx,cy) =" << cxLocal << "," << cyLocal;
        qInfo() << "半径 r =" << rLocal << "(半径std=" << stdR << ')';
        qInfo() << "相位 φ(rad)=" << phiLocal << "方向s=" << directionLocal
                << "α配准RMSE=" << rmseLocal << "rad";
        qInfo() << "z 拟合: z = " << zSlope << " * alpha + " << zIntercept;
        qInfo() << "L(3x4):\n"
                << L.at<double>(0, 0) << L.at<double>(0, 1) << L.at<double>(0, 2) << L.at<double>(0, 3) << '\n'
                << L.at<double>(1, 0) << L.at<double>(1, 1) << L.at<double>(1, 2) << L.at<double>(1, 3) << '\n'
                << L.at<double>(2, 0) << L.at<double>(2, 1) << L.at<double>(2, 2) << L.at<double>(2, 3);
    }

    return L;
}
} // namespace

/** 构造函数无特殊逻辑，仅使用默认成员值。 */
MacroMicroController::MacroMicroController(QObject *parent)
    : QObject(parent)
{
}

/**
 * @brief 绑定微动机械臂，用于像素级补偿。
 */
void MacroMicroController::attachMicroArm(MicroArm *microArm)
{
    m_microArm = microArm;
}

/**
 * @brief 绑定宏观机械臂，用于大范围移动或轨迹执行。
 */
void MacroMicroController::attachMacroArm(MacroArm *macroArm)
{
    m_macroArm = macroArm;
}

/**
 * @brief 绑定相机模块，目前主要用于保留引用。
 */
void MacroMicroController::attachCamera(CameraModule *camera)
{
    m_camera = camera;
}

/**
 * @brief 绑定图像处理模块，后续可在控制流程中获取视觉结果。
 */
void MacroMicroController::attachImageProcessor(ImageProcessor *processor)
{
    m_processor = processor;
}

/**
 * @brief 绑定俯仰姿态串口，实现 RCM 调整时的角度控制。
 */
void MacroMicroController::attachPitchSerialPort(QSerialPort *serial)
{
    m_pitchSerial = serial;
}

/**
 * @brief 更新像素到宏观坐标的映射，目前预留接口。
 */
void MacroMicroController::updateVisionToRobotMapping(const cv::Point2d &pixel,
                                                      const cv::Vec6d &macroPose)
{
    Q_UNUSED(pixel);
    Q_UNUSED(macroPose);
    qInfo() << "MacroMicroController::updateVisionToRobotMapping -> 接收到数据";
}

/**
 * @brief 根据像素误差调用微动臂进行闭环补偿。
 */
void MacroMicroController::compensateUsingMicroArm(const cv::Point2d &pixelError)
{
    if (!m_microArm) {
        qWarning() << "MacroMicroController::compensateUsingMicroArm -> 未绑定 MicroArm";
        return;
    }

    // 1. 将像素误差映射到微动臂坐标系。
    const cv::Vec2d deltaPixel(pixelError.x, pixelError.y);
    const cv::Vec2d deltaMicro = m_pixelToMicroArm * deltaPixel;

    // 2. 调用微动臂接口执行微调，Z 通道保持为 0。
    m_microArm->moveByDelta(static_cast<qint32>(deltaMicro[0]),
                             static_cast<qint32>(deltaMicro[1]), 0);

    emit motionExecuted();
}

/**
 * @brief 保留旧接口，提示改用新的 RCM 对齐流程。
 */
void MacroMicroController::performRCMMotion(const cv::Point3d &pivotInCamera,
                                            double deltaAlpha)
{
    Q_UNUSED(pivotInCamera);
    Q_UNUSED(deltaAlpha);
    qWarning() << "MacroMicroController::performRCMMotion -> 建议改用 executeRCMPitchAlignment";
}

/**
 * @brief 封装 yaw 圆轨迹拟合，直接复用匿名空间的工具函数。
 */
cv::Mat MacroMicroController::fitYawCircleMapping(const std::vector<cv::Vec4d> &pts,
                                                  double *cx,
                                                  double *cy,
                                                  double *r,
                                                  double *phi,
                                                  int *sgnOut,
                                                  double *rmseAlpha,
                                                  bool print) const
{
    return fitCircle2DGeometryThenAlpha(pts, cx, cy, r, phi, sgnOut, rmseAlpha, print);
}

/**
 * @brief 按照 RCM 约束执行俯仰对准，逐步旋转宏观臂并联动俯仰轴。
 */
bool MacroMicroController::executeRCMPitchAlignment(double alphaStartDeg,
                                                    double alphaTargetDeg,
                                                    double linkLength,
                                                    const cv::Vec6i &macroPose,
                                                    QString *errorMessage,
                                                    int minZSafety)
{
    if (!m_macroArm || !m_pitchSerial) {
        if (errorMessage) {
            *errorMessage = tr("宏观臂或俯仰轴串口未绑定");
        }
        qWarning() << "executeRCMPitchAlignment -> missing modules";
        return false;
    }

    // Step 1：根据目标方向设置俯仰角步长，0.05 度可兼顾精度与平滑性。
    const double stepDeg = (alphaTargetDeg >= alphaStartDeg) ? 0.05 : -0.05;
    const int steps = static_cast<int>(std::ceil(std::fabs(alphaTargetDeg - alphaStartDeg) /
                                                 std::fabs(stepDeg)));

    if (steps == 0) {
        return true;
    }

    // Step 2：准备俯仰轴控制报文（正向/反向）。
    const QByteArray positiveCommand = QByteArray::fromHex("3EA70104EA3200000032");
    const QByteArray negativeCommand = QByteArray::fromHex("3EA70104EACEFFFFFFCB");

    double accumulatedDx = 0.0;
    double accumulatedDz = 0.0;

    for (int i = 0; i < steps; ++i) {
        // Step 3：计算当前步和下一步的俯仰角，随后利用 RCM 几何转换成 Δx、Δz。
        const double previousDeg = alphaStartDeg + i * stepDeg;
        const double nextDeg = alphaStartDeg + (i + 1) * stepDeg;

        const double previousRad = previousDeg * kDegToRad;
        const double nextRad = nextDeg * kDegToRad;

        const double deltaX = 1000.0 * linkLength * (std::sin(nextRad) - std::sin(previousRad));
        const double deltaZ = 1000.0 * linkLength * (std::cos(nextRad) - std::cos(previousRad));

        accumulatedDx += deltaX;
        accumulatedDz += deltaZ;

        // Step 4：将 RCM 转换后的位移累加到宏观臂坐标系上。
        const int robotX = static_cast<int>(std::round(macroPose[0] + accumulatedDx));
        const int robotY = macroPose[1];
        const int robotZ = static_cast<int>(std::round(macroPose[2] + accumulatedDz));
        const int robotRx = macroPose[3];
        const int robotRy = macroPose[4];
        const int robotRz = macroPose[5];

        // Step 5：进行安全检查，如果 Z 过低则立即终止。
        if (robotZ < minZSafety) {
            const QString warning = tr("z轴位置过低碰撞");
            emit safetyWarningRaised(warning);
            if (errorMessage) {
                *errorMessage = warning;
            }
            return false;
        }

        // Step 6：下发俯仰串口命令，并让宏观臂同步到新的位姿。
        const QByteArray &command = (stepDeg > 0) ? positiveCommand : negativeCommand;
        m_pitchSerial->write(command);
        m_pitchSerial->waitForBytesWritten(300);

        sendMovePCanfd(robotX, robotY, robotZ, robotRx, robotRy, robotRz, true, 2, 800);
        spinDelay(30);

        // Step 7：响应外部的紧急停止请求。
        if (m_stopRequested) {
            m_stopRequested = false;
            break;
        }
    }

    emit motionExecuted();
    return true;
}

/**
 * @brief 执行偏航圆弧运动，并根据运动角度更新映射矩阵。
 */
bool MacroMicroController::executeYawArcMove(int currentAngleMilli,
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
                                             QString *errorMessage)
{
    if (!m_macroArm) {
        if (errorMessage) {
            *errorMessage = tr("宏观臂未绑定");
        }
        qWarning() << "executeYawArcMove -> macro arm not attached";
        return false;
    }

    if (alphaMapping.empty()) {
        if (errorMessage) {
            *errorMessage = tr("偏航映射矩阵未初始化");
        }
        qWarning() << "executeYawArcMove -> alpha mapping missing";
        return false;
    }

    // Step 1：确定起点、终点以及中间 Via 点的偏航角。
    const double alphaFinal = targetAngleMilli / 1000.0;
    const int viaAngle = (currentAngleMilli + targetAngleMilli) / 2;
    const double alphaVia = viaAngle / 1000.0;

    // Step 2：将标定矩阵转换为 double，便于矩阵乘法。
    cv::Mat alphaL;
    alphaMapping.convertTo(alphaL, CV_64F);

    // Step 3：使用 L(α) 预测 via 点和终点的笛卡尔位姿。
    cv::Mat viaVector = (cv::Mat_<double>(4, 1) << 1.0,
                         std::cos(alphaVia),
                         std::sin(alphaVia),
                         alphaVia);
    cv::Mat viaPose = alphaL * viaVector;

    const int viaX = qRound(viaPose.at<double>(0));
    const int viaY = qRound(viaPose.at<double>(1));
    const int viaZ = qRound(viaPose.at<double>(2));

    cv::Mat finalVector = (cv::Mat_<double>(4, 1) << 1.0,
                           std::cos(alphaFinal),
                           std::sin(alphaFinal),
                           alphaFinal);
    cv::Mat finalPose = alphaL * finalVector;

    const int finalX = qRound(finalPose.at<double>(0));
    const int finalY = qRound(finalPose.at<double>(1));
    const int finalZ = qRound(finalPose.at<double>(2));

    // Step 4：调用宏观臂的 movec 接口执行经由点的圆弧轨迹。
    if (!sendMoveC(viaX, viaY, viaZ, wristRx, wristRy, viaAngle,
                   finalX, finalY, finalZ, wristRx, wristRy, targetAngleMilli,
                   1)) {
        if (errorMessage) {
            *errorMessage = tr("发送圆弧轨迹指令失败");
        }
        return false;
    }

    // Step 5：更新在线 homography，使视觉映射与偏航同步。
    const double beta = alphaFinal - alphaCalibrationRad;
    updatedHomography = rotateTargetAboutUvCenter(homographyCalib, beta, imageCenterU, imageCenterV);
    updatedLinear = cv::Matx22d(updatedHomography.at<double>(0, 0), updatedHomography.at<double>(0, 1),
                                updatedHomography.at<double>(1, 0), updatedHomography.at<double>(1, 1));

    emit motionExecuted();
    return true;
}

/**
 * @brief 将标定 Homography 绕像面中心旋转 beta 角度。
 */
cv::Mat MacroMicroController::rotateTargetAboutUvCenter(const cv::Mat &H_in,
                                                        double beta,
                                                        double u0,
                                                        double v0) const
{
    CV_Assert(H_in.rows == 3 && H_in.cols == 3);
    cv::Mat H;
    H_in.convertTo(H, CV_64F);

    // 1. 拆分 Homography 的线性部分和平移部分。
    cv::Matx22d M(H.at<double>(0, 0), H.at<double>(0, 1),
                  H.at<double>(1, 0), H.at<double>(1, 1));
    cv::Vec2d t(H.at<double>(0, 2), H.at<double>(1, 2));

    // 2. 构建绕中心旋转的 2D 旋转矩阵。
    const double cb = std::cos(beta);
    const double sb = std::sin(beta);
    cv::Matx22d R(cb, -sb,
                  sb,  cb);
    cv::Vec2d center(u0, v0);

    // 3. 按照 H' = R*M, t' = t + (I - R)*M*c 推导更新。
    const cv::Matx22d identity = cv::Matx22d::eye();
    const cv::Matx22d rotated = R * M;
    const cv::Vec2d translated = t + (identity - R) * (M * center);

    return (cv::Mat_<double>(3, 3)
            << rotated(0, 0), rotated(0, 1), translated[0],
               rotated(1, 0), rotated(1, 1), translated[1],
               0.0, 0.0, 1.0);
}

/**
 * @brief 设置停止标志，使长循环在下一次检查时退出。
 */
void MacroMicroController::requestStop()
{
    m_stopRequested = true;
}

/**
 * @brief 通过宏观臂接口发送 movep_canfd 指令。
 */
bool MacroMicroController::sendMovePCanfd(int x, int y, int z, int rx, int ry, int rz,
                                          bool follow, int trajectoryMode, int radio)
{
    if (!m_macroArm) {
        qWarning() << "sendMovePCanfd -> macro arm not attached";
        return false;
    }

    // 构造 movep_canfd 的 JSON 指令，包含目标姿态与跟随模式等参数。
    QJsonObject cmd;
    cmd["command"] = "movep_canfd";
    cmd["pose"] = QJsonArray{x, y, z, rx, ry, rz};
    cmd["follow"] = follow ? 1 : 0;
    cmd["trajectory_mode"] = trajectoryMode;
    cmd["radio"] = radio;

    const QString payload = QString::fromUtf8(QJsonDocument(cmd).toJson(QJsonDocument::Compact));
    m_macroArm->sendCommand(payload);
    return true;
}

/**
 * @brief 通过宏观臂接口发送 movec 圆弧指令。
 */
bool MacroMicroController::sendMoveC(int via_x, int via_y, int via_z, int via_rx, int via_ry, int via_rz,
                                     int x, int y, int z, int rx, int ry, int rz,
                                     int v, int r, int loop, bool trajectoryConnect)
{
    if (!m_macroArm) {
        qWarning() << "sendMoveC -> macro arm not attached";
        return false;
    }

    // 构造 movec 指令的 JSON 结构，包括 via 点和目标点两个姿态。
    QJsonObject cmd;
    cmd["command"] = "movec";

    QJsonArray viaPose = QJsonArray{via_x, via_y, via_z, via_rx, via_ry, via_rz};
    QJsonArray toPose = QJsonArray{x, y, z, rx, ry, rz};

    QJsonObject poseObj;
    poseObj["pose_via"] = viaPose;
    poseObj["pose_to"] = toPose;
    cmd["pose"] = poseObj;

    cmd["v"] = v;
    cmd["r"] = r;
    cmd["loop"] = loop;
    cmd["trajectory_connect"] = trajectoryConnect ? 1 : 0;

    const QString payload = QString::fromUtf8(QJsonDocument(cmd).toJson(QJsonDocument::Compact));
    m_macroArm->sendCommand(payload);
    return true;
}

/**
 * @brief 在保持事件循环响应的前提下延时指定毫秒。
 */
void MacroMicroController::spinDelay(int milliseconds)
{
    if (milliseconds <= 0) {
        return;
    }

    // 使用 QElapsedTimer 实现非阻塞延时，并驱动 Qt 事件循环。
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < milliseconds) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 5);
    }
}
