#include "micro_arm.h"

#include <QDebug>

MicroArm::MicroArm(QObject *parent)
    : QObject(parent)
{
    // 将 readyRead 信号连接到自定义槽，确保串口数据能够第一时间被处理。
    connect(&m_serial, &QSerialPort::readyRead, this, &MicroArm::handleReadyRead);
}

bool MicroArm::applySerialSettings(const QString &portName,
                                   const QString &baudRateText,
                                   const QString &parityText,
                                   const QString &dataBitsText,
                                   const QString &stopBitsText,
                                   QString *errorMessage)
{
    SerialConfig cfg;
    cfg.portName = portName;

    bool conversionOk = true;
    cfg.baudRate = baudRateText.toInt(&conversionOk);
    if (!conversionOk) {
        if (errorMessage) {
            *errorMessage = tr("无法识别的波特率: %1").arg(baudRateText);
        }
        return false;
    }

    if (parityText.compare(QStringLiteral("None"), Qt::CaseInsensitive) == 0) {
        cfg.parity = QSerialPort::NoParity;
    } else if (parityText.compare(QStringLiteral("Even"), Qt::CaseInsensitive) == 0) {
        cfg.parity = QSerialPort::EvenParity;
    } else if (parityText.compare(QStringLiteral("Odd"), Qt::CaseInsensitive) == 0) {
        cfg.parity = QSerialPort::OddParity;
    } else {
        if (errorMessage) {
            *errorMessage = tr("无法识别的校验位: %1").arg(parityText);
        }
        return false;
    }

    if (dataBitsText == QLatin1String("8")) {
        cfg.dataBits = QSerialPort::Data8;
    } else if (dataBitsText == QLatin1String("7")) {
        cfg.dataBits = QSerialPort::Data7;
    } else if (dataBitsText == QLatin1String("6")) {
        cfg.dataBits = QSerialPort::Data6;
    } else if (dataBitsText == QLatin1String("5")) {
        cfg.dataBits = QSerialPort::Data5;
    } else {
        if (errorMessage) {
            *errorMessage = tr("无法识别的数据位设置: %1").arg(dataBitsText);
        }
        return false;
    }

    if (stopBitsText == QLatin1String("1")) {
        cfg.stopBits = QSerialPort::OneStop;
    } else if (stopBitsText == QLatin1String("1.5")) {
        cfg.stopBits = QSerialPort::OneAndHalfStop;
    } else if (stopBitsText == QLatin1String("2")) {
        cfg.stopBits = QSerialPort::TwoStop;
    } else {
        if (errorMessage) {
            *errorMessage = tr("无法识别的停止位设置: %1").arg(stopBitsText);
        }
        return false;
    }

    configureSerialPort(cfg);
    return true;
}

void MicroArm::configureSerialPort(const SerialConfig &config)
{
    m_config = config;
    m_serial.setPortName(m_config.portName);
    m_serial.setBaudRate(m_config.baudRate);
    m_serial.setDataBits(m_config.dataBits);
    m_serial.setParity(m_config.parity);
    m_serial.setStopBits(m_config.stopBits);
    m_serial.setFlowControl(m_config.flowControl);
}

bool MicroArm::openSerialPort()
{
    if (m_serial.isOpen()) {
        m_serial.close();
    }

    if (!m_serial.open(QIODevice::ReadWrite)) {
        qWarning() << "MicroArm::openSerialPort -> 无法打开串口" << m_serial.portName()
                   << ", error =" << m_serial.errorString();
        return false;
    }

    qInfo() << "MicroArm::openSerialPort -> 打开串口成功" << m_serial.portName();
    return true;
}

bool MicroArm::openConfiguredPort(QString *errorMessage)
{
    if (openSerialPort()) {
        return true;
    }

    if (errorMessage) {
        *errorMessage = m_serial.errorString();
    }
    return false;
}

void MicroArm::closeSerialPort()
{
    if (!m_serial.isOpen()) {
        return;
    }

    m_serial.close();
    qInfo() << "MicroArm::closeSerialPort -> 串口已关闭" << m_serial.portName();
}

void MicroArm::closePort()
{
    closeSerialPort();
}

bool MicroArm::isSerialPortOpen() const
{
    return m_serial.isOpen();
}

qint64 MicroArm::sendRawCommand(const QByteArray &payload)
{
    if (!m_serial.isOpen()) {
        qWarning() << "MicroArm::sendRawCommand -> 串口未打开，无法发送数据";
        return -1;
    }

    const qint64 bytesWritten = m_serial.write(payload);
    if (bytesWritten == -1) {
        qWarning() << "MicroArm::sendRawCommand -> 写入失败" << m_serial.errorString();
    } else if (!m_serial.waitForBytesWritten(10)) {
        qWarning() << "MicroArm::sendRawCommand -> 写入超时";
    }

    return bytesWritten;
}

bool MicroArm::moveToPose(qint32 x, qint32 y, qint32 z)
{
    const QByteArray frame = buildMoveFrame(x, y, z);
    if (!sendFrame(frame, QStringLiteral("MoveToPose"))) {
        return false;
    }

    m_currentPose = cv::Vec3i(x, y, z);
    emit poseUpdated(m_currentPose);
    return true;
}

bool MicroArm::moveByDelta(qint32 deltaX, qint32 deltaY, qint32 deltaZ)
{
    const cv::Vec3i newPose = m_currentPose + cv::Vec3i(deltaX, deltaY, deltaZ);
    return moveToPose(newPose[0], newPose[1], newPose[2]);
}

void MicroArm::requestEmergencyStop()
{
    QByteArray frame;
    frame.append(static_cast<char>(0xAB)); // 假设 0xAB 为紧急停止命令。
    frame.append(static_cast<char>(0xBA));
    sendRawCommand(frame);
}

bool MicroArm::requestCurrentPosition()
{
    const QByteArray frame = QByteArray::fromHex("630D");
    return sendFrame(frame, QStringLiteral("RequestCurrentPosition"));
}

bool MicroArm::setVelocityAndResolution(int speed, const QString &resolutionText, QString *errorMessage)
{
    const bool highResolution = !(resolutionText.compare(QStringLiteral("0.2"), Qt::CaseInsensitive) == 0);
    const QByteArray frame = buildVelocityResolutionFrame(speed, highResolution, errorMessage);
    if (frame.isEmpty()) {
        return false;
    }
    return sendFrame(frame, QStringLiteral("SetVelocityResolution"));
}

bool MicroArm::setMoveMode(MoveMode mode)
{
    const QByteArray frame = (mode == MoveMode::Relative)
                                 ? QByteArray::fromHex("620D")
                                 : QByteArray::fromHex("610D");
    return sendFrame(frame, QStringLiteral("SetMoveMode"));
}

bool MicroArm::zeroCurrentPosition()
{
    const QByteArray frame = QByteArray::fromHex("6F0D");
    if (!sendFrame(frame, QStringLiteral("ZeroCurrentPosition"))) {
        return false;
    }

    m_currentPose = cv::Vec3i(0, 0, 0);
    emit poseUpdated(m_currentPose);
    return true;
}

bool MicroArm::interruptMotion()
{
    const QByteArray frame = QByteArray::fromHex("03");
    return sendFrame(frame, QStringLiteral("InterruptMotion"));
}

bool MicroArm::sendManualHexCommand(const QString &hexString, QString *errorMessage)
{
    QString sanitized = hexString;
    sanitized.remove(QLatin1Char(' '));
    sanitized.remove(QLatin1Char('\n'));
    sanitized.remove(QLatin1Char('\r'));

    if (sanitized.isEmpty() || (sanitized.size() % 2) != 0) {
        if (errorMessage) {
            *errorMessage = tr("十六进制命令长度必须为偶数，且不可为空");
        }
        return false;
    }

    const QByteArray frame = QByteArray::fromHex(sanitized.toUtf8());
    return sendFrame(frame, QStringLiteral("ManualHexCommand"));
}

cv::Vec3i MicroArm::currentPose() const
{
    return m_currentPose;
}

QSerialPort &MicroArm::serialPort()
{
    return m_serial;
}

const QSerialPort &MicroArm::serialPort() const
{
    return m_serial;
}

void MicroArm::handleReadyRead()
{
    const QByteArray raw = m_serial.readAll();
    if (raw.isEmpty()) {
        return;
    }

    const QString hex = toHexString(raw);
    emit serialDataArrived(raw, hex);
}

QString MicroArm::toHexString(const QByteArray &raw) const
{
    QString hex;
    for (unsigned char byte : raw) {
        hex.append(QString::asprintf("%02X ", byte));
    }
    return hex.trimmed();
}

bool MicroArm::sendFrame(const QByteArray &frame, const QString &context) const
{
    if (frame.isEmpty()) {
        qWarning() << "MicroArm::sendFrame ->" << context << "帧为空，已忽略";
        return false;
    }

    const qint64 written = const_cast<MicroArm*>(this)->sendRawCommand(frame);
    if (written == -1) {
        qWarning() << "MicroArm::sendFrame ->" << context << "写入失败";
        return false;
    }

    return true;
}

QByteArray MicroArm::buildMoveFrame(qint32 x, qint32 y, qint32 z) const
{
    QByteArray frame;
    frame.reserve(1 + 4 * 3 + 1);
    frame.append(static_cast<char>(0x6D));

    auto appendLittleEndian = [&frame](qint32 value) {
        const quint32 raw = static_cast<quint32>(value);
        for (int i = 0; i < 4; ++i) {
            frame.append(static_cast<char>((raw >> (i * 8)) & 0xFF));
        }
    };

    appendLittleEndian(x);
    appendLittleEndian(y);
    appendLittleEndian(-z); // Z 轴方向取反后下发，XY 轴保持不变。
    frame.append(static_cast<char>(0x0D));

    return frame;
}

QByteArray MicroArm::buildVelocityResolutionFrame(int speed, bool highResolution, QString *errorMessage) const
{
    if (speed < 0 || speed > 0x7FFF) {
        if (errorMessage) {
            *errorMessage = tr("速度数值超出允许范围 (0-32767): %1").arg(speed);
        }
        return {};
    }

    const int resolutionBit = highResolution ? 1 : 0;
    const int payload = (resolutionBit << 15) | (speed & 0x7FFF);

    QByteArray frame;
    frame.reserve(1 + 2 + 1);
    frame.append(static_cast<char>(0x56));
    frame.append(static_cast<char>(payload & 0xFF));
    frame.append(static_cast<char>((payload >> 8) & 0xFF));
    frame.append(static_cast<char>(0x0D));
    return frame;
}
