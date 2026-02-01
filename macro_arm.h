#ifndef MACRO_ARM_H
#define MACRO_ARM_H

#include <QObject>
#include <QTcpSocket>
#include <QHostAddress>
#include <QJsonObject>
#include <opencv2/core.hpp>

/**
 * @class MacroArm
 * @brief 宏观机械臂（大型六轴机械臂）通信模块。
 *
 * 该类封装与宏观机械臂控制柜之间的 TCP 通信逻辑，负责：
 *  - 管理命令通道 socket 的连接状态；
 *  - 发送 MoveL/MoveJ 等运动指令；
 *  - 接收并转发来自控制柜的状态反馈；
 *  - 缓存当前的末端位姿，便于上层查询与 UI 显示。
 */
class MacroArm : public QObject
{
    Q_OBJECT
public:
    struct NetworkConfig
    {
        QHostAddress host = QHostAddress::LocalHost; ///< 控制柜 IP 地址。
        quint16 commandPort = 8080;                  ///< 命令通道端口。
        // quint16 feedbackPort = 8089;              ///< ⚠️ 反馈通道端口暂未启用，保留旧注释以便后续恢复。
    };

    explicit MacroArm(QObject *parent = nullptr);

    /**
     * @brief 更新网络配置，未主动触发连接。
     */
    void configureNetwork(const NetworkConfig &config);

    /**
     * @brief 结合 IP/端口信息直接配置并尝试连接控制柜。
     *
     * @param ipAddress     点分十进制格式的机器人控制柜 IP。
     * @param commandPort   命令通道端口号。
     * @param errorMessage  可选的错误输出；当连接失败时写入失败原因。
     * @return true 表示配置成功且成功建立连接，false 表示发生错误。
     */
    bool configureAndConnect(const QString &ipAddress,
                             quint16 commandPort,
                             QString *errorMessage = nullptr);

    /**
     * @brief 尝试建立与控制柜的 TCP 连接。
     * @return 成功返回 true，否则返回 false。
     */
    bool connectToRobot();

    /**
     * @brief 尝试建立连接，同时提供失败原因。
     */
    bool connectToRobot(QString *errorMessage);

    /**
     * @brief 主动断开与控制柜的连接。
     */
    void disconnectFromRobot();

    /**
     * @brief 判断命令端口是否已经连接。
     */
    bool isConnected() const;

    /**
     * @brief 通过命令端口发送一条纯文本指令。
     */
    void sendCommand(const QString &command);

    /**
     * @brief 将当前末端位姿缓存暴露给外部。
     */
    cv::Vec6d currentPose() const;

    /**
     * @brief 发送一条通用 JSON 指令到控制柜。
     * @param json  需发送的 JSON 对象。
     * @param contextHint  调试用上下文描述，会打印到日志中。
     */
    void sendJsonCommand(const QJsonObject &json, const QString &contextHint = {});

    /**
     * @brief 下发直线插补命令（MoveL）。
     */
    void moveL(int x, int y, int z, int rx, int ry, int rz,
               int velocity, int blendRadius = 10, bool trajectoryConnect = false);

    /**
     * @brief 下发圆弧插补命令（MoveC）。
     */
    void moveC(int via_x, int via_y, int via_z, int via_rx, int via_ry, int via_rz,
               int x, int y, int z, int rx, int ry, int rz,
               int velocity, int blendRadius = 0, int loop = 0, bool trajectoryConnect = false);

    /**
     * @brief 下发 movep_canfd 命令。
     */
    void movePCanfd(int x, int y, int z, int rx, int ry, int rz,
                    bool follow, int trajectoryMode, int ratio);

    /**
     * @brief 设置单轴步进运动命令。
     */
    void setPosStep(const QString &axisStep, int steps, int velocity);

    /**
     * @brief 控制机械臂上电/下电。
     */
    void setArmPower(bool enabled);

    /**
     * @brief 请求机械臂停止或减速停止。
     */
    void requestArmStop(bool slowStop);

    /**
     * @brief 查询机械臂当前状态。
     */
    void requestCurrentState();

    /**
     * @brief 设置线速度上限。
     */
    void setMaxLineSpeed(int speed);

    /**
     * @brief 设置线加速度上限。
     */
    void setMaxLineAcceleration(int acceleration);

signals:
    void connected();
    void disconnected();

    /**
     * @brief 控制柜反馈的原始文本。
     */
    void feedbackReceived(const QString &text);

    /**
     * @brief 末端位姿更新时发出的信号。
     */
    void poseUpdated(const cv::Vec6d &pose);

private slots:
    void handleCommandReadyRead();
    void handleSocketError(QAbstractSocket::SocketError error);

private:
    /**
     * @brief 在收到反馈报文时解析末端位姿。
     *
     * 该函数仅给出示例解析流程，实际格式需依据控制柜的真实协议进行调整。
     */
    void tryParsePoseFromFeedback(const QString &text);

    QTcpSocket m_commandSocket;  ///< 下发运动命令的 TCP 通道。
    NetworkConfig m_config;      ///< 最近一次的网络配置。
    cv::Vec6d m_currentPose{0, 0, 0, 0, 0, 0}; ///< 缓存末端位姿 (x,y,z,rx,ry,rz)。
};

#endif // MACRO_ARM_H
