#include "macro_arm.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>

MacroArm::MacroArm(QObject *parent)
    : QObject(parent)
{
    connect(&m_commandSocket, &QTcpSocket::readyRead,
            this, &MacroArm::handleCommandReadyRead);

    const auto errorHandler = [this](QAbstractSocket::SocketError error) {
        Q_UNUSED(error);
        qWarning() << "MacroArm -> socket error:" << m_commandSocket.errorString();
        emit disconnected();
    };

    connect(&m_commandSocket, &QTcpSocket::errorOccurred, this, errorHandler);
}

void MacroArm::configureNetwork(const NetworkConfig &config)
{
    m_config = config;
}

bool MacroArm::configureAndConnect(const QString &ipAddress,
                                   quint16 commandPort,
                                   QString *errorMessage)
{
    QHostAddress parsedAddress;
    if (!parsedAddress.setAddress(ipAddress)) {
        if (errorMessage) {
            *errorMessage = tr("无效的控制柜 IP 地址: %1").arg(ipAddress);
        }
        return false;
    }

    NetworkConfig cfg;
    cfg.host = parsedAddress;
    cfg.commandPort = commandPort;
    configureNetwork(cfg);

    return connectToRobot(errorMessage);
}

bool MacroArm::connectToRobot()
{
    if (isConnected()) {
        return true;
    }

    m_commandSocket.abort();

    m_commandSocket.connectToHost(m_config.host, m_config.commandPort);
    if (!m_commandSocket.waitForConnected(3000)) {
        qWarning() << "MacroArm::connectToRobot -> 命令端口连接失败"
                   << m_commandSocket.errorString();
        return false;
    }

    emit connected();
    return true;
}

bool MacroArm::connectToRobot(QString *errorMessage)
{
    if (connectToRobot()) {
        return true;
    }

    if (errorMessage) {
        *errorMessage = tr("无法建立与控制柜的 TCP 连接（IP: %1, CmdPort: %2）")
                            .arg(m_config.host.toString())
                            .arg(m_config.commandPort);
    }
    return false;
}

void MacroArm::disconnectFromRobot()
{
    m_commandSocket.disconnectFromHost();
    emit disconnected();
}

bool MacroArm::isConnected() const
{
    return m_commandSocket.state() == QAbstractSocket::ConnectedState;
}

void MacroArm::sendCommand(const QString &command)
{
    if (!isConnected()) {
        qWarning() << "MacroArm::sendCommand -> 未连接到控制柜";
        return;
    }

    const QByteArray bytes = command.toUtf8();
    m_commandSocket.write(bytes);
    m_commandSocket.flush();
}

cv::Vec6d MacroArm::currentPose() const
{
    return m_currentPose;
}

void MacroArm::sendJsonCommand(const QJsonObject &json, const QString &contextHint)
{
    const QJsonDocument doc(json);
    const QString payload = QString::fromUtf8(doc.toJson(QJsonDocument::Compact));
    sendCommand(payload);

    if (!contextHint.isEmpty()) {
        qInfo() << "MacroArm::sendJsonCommand ->" << contextHint << payload;
    }
}

void MacroArm::moveL(int x, int y, int z, int rx, int ry, int rz,
                     int velocity, int blendRadius, bool trajectoryConnect)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("movel");
    cmd["pose"] = QJsonArray{ x, y, z, rx, ry, rz };
    cmd["v"] = velocity;
    cmd["r"] = blendRadius;
    cmd["trajectory_connect"] = trajectoryConnect ? 1 : 0;
    sendJsonCommand(cmd, QStringLiteral("MoveL"));
}

void MacroArm::moveC(int via_x, int via_y, int via_z, int via_rx, int via_ry, int via_rz,
                     int x, int y, int z, int rx, int ry, int rz,
                     int velocity, int blendRadius, int loop, bool trajectoryConnect)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("movec");

    QJsonObject poseObject;
    poseObject["pose_via"] = QJsonArray{ via_x, via_y, via_z, via_rx, via_ry, via_rz };
    poseObject["pose_to"] = QJsonArray{ x, y, z, rx, ry, rz };
    cmd["pose"] = poseObject;

    cmd["v"] = velocity;
    cmd["r"] = blendRadius;
    cmd["loop"] = loop;
    cmd["trajectory_connect"] = trajectoryConnect ? 1 : 0;

    sendJsonCommand(cmd, QStringLiteral("MoveC"));
}

void MacroArm::movePCanfd(int x, int y, int z, int rx, int ry, int rz,
                          bool follow, int trajectoryMode, int ratio)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("movep_canfd");
    cmd["pose"] = QJsonArray{ x, y, z, rx, ry, rz };
    cmd["follow"] = follow ? 1 : 0;
    cmd["trajectory_mode"] = trajectoryMode;
    cmd["radio"] = ratio;

    sendJsonCommand(cmd, QStringLiteral("MoveP_CANFD"));
}

void MacroArm::setPosStep(const QString &axisStep, int steps, int velocity)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("set_pos_step");
    cmd["step_type"] = axisStep;
    cmd["step"] = steps;
    cmd["v"] = velocity;

    sendJsonCommand(cmd, QStringLiteral("SetPosStep"));
}

void MacroArm::setArmPower(bool enabled)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("set_arm_power");
    cmd["arm_power"] = enabled ? 1 : 0;
    sendJsonCommand(cmd, QStringLiteral("SetArmPower"));
}

void MacroArm::requestArmStop(bool slowStop)
{
    QJsonObject cmd;
    cmd["command"] = slowStop ? QStringLiteral("set_arm_slow_stop")
                               : QStringLiteral("set_arm_stop");
    sendJsonCommand(cmd, slowStop ? QStringLiteral("ArmSlowStop") : QStringLiteral("ArmStop"));
}

void MacroArm::requestCurrentState()
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("get_current_arm_state");
    sendJsonCommand(cmd, QStringLiteral("GetCurrentState"));
}

void MacroArm::setMaxLineSpeed(int speed)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("set_arm_max_line_speed");
    cmd["arm_line_speed"] = speed;
    sendJsonCommand(cmd, QStringLiteral("SetMaxLineSpeed"));
}

void MacroArm::setMaxLineAcceleration(int acceleration)
{
    QJsonObject cmd;
    cmd["command"] = QStringLiteral("set_arm_max_line_acc");
    cmd["arm_line_acc"] = acceleration;
    sendJsonCommand(cmd, QStringLiteral("SetMaxLineAcceleration"));
}

void MacroArm::handleCommandReadyRead()
{
    const QString text = QString::fromUtf8(m_commandSocket.readAll());
    if (text.isEmpty()) {
        return;
    }
    emit feedbackReceived(text);
    tryParsePoseFromFeedback(text);
}

void MacroArm::handleSocketError(QAbstractSocket::SocketError error)
{
    Q_UNUSED(error);
    qWarning() << "MacroArm::handleSocketError ->" << m_commandSocket.errorString();
    emit disconnected();
}

void MacroArm::tryParsePoseFromFeedback(const QString &text)
{
    // 示例实现：假设反馈报文形如 "POSE x y z rx ry rz"。
    const QStringList parts = text.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (parts.size() < 7 || parts.first() != QLatin1String("POSE")) {
        return;
    }

    bool ok = true;
    cv::Vec6d pose;
    for (int i = 0; i < 6; ++i) {
        pose[i] = parts[i + 1].toDouble(&ok);
        if (!ok) {
            return;
        }
    }

    m_currentPose = pose;
    emit poseUpdated(m_currentPose);
}
