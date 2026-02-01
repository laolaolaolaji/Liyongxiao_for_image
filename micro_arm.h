#ifndef MICRO_ARM_H
#define MICRO_ARM_H

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QTimer>
#include <QString>
#include <opencv2/core.hpp>

/**
 * @class MicroArm
 * @brief 微动机械臂（显微操作平台）驱动模块。
 *
 * 该类负责维护串口连接、发送指令、解析反馈以及保存当前微动臂的状态。通过
 * 将与微动臂相关的逻辑集中在一起，可以显著降低 QMainWindow 的复杂度。
 *
 * 使用方式：
 *  1. 在主线程中实例化 MicroArm；
 *  2. 调用 configureSerialPort() 设置串口参数；
 *  3. 调用 openSerialPort()/closeSerialPort() 控制连接；
 *  4. 通过 moveByDelta()/moveToPose() 等接口发送位姿命令；
 *  5. 监听 serialDataArrived() 信号以获取实时反馈。
 */
class MicroArm : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 保存串口配置的结构体。
     *
     * 相比直接传递多个参数，结构体的设计使得调用端代码更加清晰直观，
     * 同时也方便在将来扩展新的配置项（例如校验位、自定义协议参数等）。
     */
    struct SerialConfig
    {
        QString portName;                 ///< 串口号，例如 "COM3"。
        qint32 baudRate = QSerialPort::Baud115200; ///< 波特率。
        QSerialPort::DataBits dataBits = QSerialPort::Data8; ///< 数据位长度。
        QSerialPort::Parity parity = QSerialPort::NoParity;   ///< 奇偶校验类型。
        QSerialPort::StopBits stopBits = QSerialPort::OneStop;///< 停止位长度。
        QSerialPort::FlowControl flowControl = QSerialPort::NoFlowControl; ///< 流控方式。
    };

    explicit MicroArm(QObject *parent = nullptr);

    /**
     * @brief 根据 UI 字符串配置串口参数。
     *
     * 与原始的 configureSerialPort 区别在于，该接口负责把字符串（如 "9600"、"Even"）
     * 转换为 QSerialPort 所需的枚举，并在失败时返回错误原因，避免 UI 层反复堆砌判断逻辑。
     */
    bool applySerialSettings(const QString &portName,
                             const QString &baudRateText,
                             const QString &parityText,
                             const QString &dataBitsText,
                             const QString &stopBitsText,
                             QString *errorMessage = nullptr);

    /**
     * @brief 应用新的串口配置。
     *
     * 该函数仅保存配置，不会立即打开串口。实际的打开动作由 openSerialPort() 执行。
     */
    void configureSerialPort(const SerialConfig &config);

    /**
     * @brief 尝试打开串口并建立与微动臂的通信连接。
     * @return true 表示成功，false 表示失败。
     */
    bool openSerialPort();

    /**
     * @brief 打开串口并返回错误描述。
     */
    bool openConfiguredPort(QString *errorMessage = nullptr);

    /**
     * @brief 主动关闭串口连接。
     */
    void closeSerialPort();

    /**
     * @brief 统一封装的关闭接口，便于语义化调用。
     */
    void closePort();

    /**
     * @brief 当前串口是否处于打开状态。
     */
    bool isSerialPortOpen() const;

    /**
     * @brief 直接向串口发送裸数据帧。
     * @param payload 待发送的字节数组。
     * @return 实际写入的字节数。
     */
    qint64 sendRawCommand(const QByteArray &payload);

    /**
     * @brief 微动臂按绝对坐标移动。
     *
     * 这里的实现仅封装了常见的 "移动到目标位置" 场景；
     * 如果协议需要更复杂的帧格式，可以在内部生成完整报文。
     */
    bool moveToPose(qint32 x, qint32 y, qint32 z);

    /**
     * @brief 根据增量值进行相对位移。
     */
    bool moveByDelta(qint32 deltaX, qint32 deltaY, qint32 deltaZ);

    /**
     * @brief 请求立即停止当前动作。
     */
    void requestEmergencyStop();

    /**
     * @brief 请求微动臂反馈当前位置。
     */
    bool requestCurrentPosition();

    /**
     * @brief 设置速度和分辨率。
     */
    bool setVelocityAndResolution(int speed, const QString &resolutionText, QString *errorMessage = nullptr);

    /**
     * @brief 设置运动模式（绝对/相对）。
     */
    enum class MoveMode { Absolute, Relative };
    bool setMoveMode(MoveMode mode);

    /**
     * @brief 将三个轴当前位置清零。
     */
    bool zeroCurrentPosition();

    /**
     * @brief 打断当前运动。
     */
    bool interruptMotion();

    /**
     * @brief 直接下发一段十六进制命令字符串（调试用途）。
     */
    bool sendManualHexCommand(const QString &hexString, QString *errorMessage = nullptr);

    /**
     * @brief 读取当前缓存的末端位姿。
     * @return 以 cv::Vec3i 形式返回 (x,y,z)。
     */
    cv::Vec3i currentPose() const;

    /**
     * @brief 暴露底层串口引用，方便遗留代码逐步迁移。
     */
    QSerialPort &serialPort();
    const QSerialPort &serialPort() const;

signals:
    /**
     * @brief 串口收到完整数据包时发出的信号。
     *
     * @param rawBytes  原始字节内容，便于上层自行解析。
     * @param readableHex 经过格式化的十六进制字符串，方便在调试界面展示。
     */
    void serialDataArrived(const QByteArray &rawBytes, const QString &readableHex);

    /**
     * @brief 微动臂位姿发生改变时发出的信号。
     */
    void poseUpdated(const cv::Vec3i &pose);

private slots:
    /**
     * @brief Qt 串口 readyRead() 信号触发后的读取逻辑。
     */
    void handleReadyRead();

private:
    bool sendFrame(const QByteArray &frame, const QString &context) const;
    QByteArray buildMoveFrame(qint32 x, qint32 y, qint32 z) const;
    QByteArray buildVelocityResolutionFrame(int speed, bool highResolution, QString *errorMessage) const;

    /**
     * @brief 将原始字节数据转换为大写十六进制字符串。
     */
    QString toHexString(const QByteArray &raw) const;

    QSerialPort m_serial;    ///< 实际的串口对象。
    SerialConfig m_config;   ///< 最近一次的串口配置缓存。
    cv::Vec3i m_currentPose{0, 0, 0}; ///< 缓存当前位姿，用于外部查询。
};

#endif // MICRO_ARM_H
