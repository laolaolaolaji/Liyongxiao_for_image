#include "JsonExplain.h"

JsonExplain::JsonExplain(QObject *parent)
    : QObject(parent)
{
}

void JsonExplain::JsonGetCurrentState(const QString &messageReceived)
{
    // Step 1: 对原始文本做基础归一化处理。
    //  - messageReceived 可能来源于 QTextEdit/QPlainTextEdit，在保存为字符串时，"\r\n" 会以
    //    文本形式保留，因此需要将反斜杠序列还原为真正的换行符，避免影响后续分割。
    //  - 同时去除两端的空白字符，如果仍旧为空则无需继续解析。
    QString normalized = messageReceived;
    normalized.replace(QStringLiteral("\\r\\n"), QStringLiteral("\n"));
    normalized.replace(QStringLiteral("\\r"), QStringLiteral("\n"));
    normalized.replace(QStringLiteral("\\n"), QStringLiteral("\n"));
    normalized.replace(QChar('\r'), QChar('\n'));

    const QString trimmedPayload = normalized.trimmed();
    if (trimmedPayload.isEmpty()) {
        qWarning() << "JSON 解析失败 -> 空字符串";
        return;
    }

    // Step 2: 将归一化后的内容按行切分，每一段都可能是一个完整的 JSON 报文。
    QStringList fragments = trimmedPayload.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    if (fragments.isEmpty()) {
        fragments.append(trimmedPayload);
    }

    bool parsedSuccessfully = false;
    QString lastError;

    for (QString fragment : fragments) {
        fragment = fragment.trimmed();
        if (fragment.isEmpty()) {
            continue;
        }

        // Step 3: 某些反馈报文前后可能带有提示信息或其他非 JSON 字符，
        //         尝试截取第一个 '{' 与最后一个 '}' 之间的部分作为候选 JSON。
        const int beginIndex = fragment.indexOf(QLatin1Char('{'));
        const int endIndex = fragment.lastIndexOf(QLatin1Char('}'));
        if (beginIndex != -1 && endIndex != -1 && endIndex >= beginIndex) {
            fragment = fragment.mid(beginIndex, endIndex - beginIndex + 1);
        }

        if (fragment.isEmpty()) {
            continue;
        }

        // Step 4: 针对形如 "{...}\r" 的场景，移除末尾的回车符。
        if (fragment.endsWith('\r')) {
            fragment.chop(1);
        }

        QJsonParseError parseError;
        const QJsonDocument jsonDoc = QJsonDocument::fromJson(fragment.toUtf8(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !jsonDoc.isObject()) {
            lastError = parseError.errorString();
            continue;
        }

        parsedSuccessfully = true;
        const QJsonObject jsonObj = jsonDoc.object();

        // Step 3: 提取 "state" 字段用于状态显示。
        const QString state = jsonObj.value("state").toString();
        if (!state.isEmpty()) {
            emit stateParsed(state);
        }

        if (jsonObj.contains("arm_state") && jsonObj.value("arm_state").isObject()) {
            const QJsonObject armState = jsonObj.value("arm_state").toObject();

            // joint
            QList<int> jointList;
            if (armState.contains("joint") && armState.value("joint").isArray()) {
                const QJsonArray jointArray = armState.value("joint").toArray();
                for (const QJsonValue &val : jointArray) {
                    jointList.append(val.toInt());
                }
                if (!jointList.isEmpty()) {
                    emit jointAnglesParsed(jointList);
                }
            }

            // pose
            QList<int> poseList;
            if (armState.contains("pose") && armState.value("pose").isArray()) {
                const QJsonArray poseArray = armState.value("pose").toArray();
                for (const QJsonValue &val : poseArray) {
                    poseList.append(val.toInt());
                }
                if (!poseList.isEmpty()) {
                    emit poseParsed(poseList);
                }
            }

            // 错误信息：旧协议是 "err" 数组，新协议提供 "arm_err" 与 "sys_err"。
            QList<int> errList;
            if (armState.contains("err") && armState.value("err").isArray()) {
                const QJsonArray errArray = armState.value("err").toArray();
                for (const QJsonValue &val : errArray) {
                    errList.append(val.toInt());
                }
            } else {
                if (armState.contains("arm_err")) {
                    errList.append(armState.value("arm_err").toInt());
                }
                if (armState.contains("sys_err")) {
                    errList.append(armState.value("sys_err").toInt());
                }
            }

            if (!errList.isEmpty()) {
                emit errorsParsed(errList);
            }
        }

        // 成功解析其中一个片段即可，无需继续遍历。
        break;
    }

    if (!parsedSuccessfully) {
        qWarning() << "JSON 解析失败" << trimmedPayload << "error:" << lastError;
    }
}
