#ifndef JSONEXPLAIN_H
#define JSONEXPLAIN_H

#include <QObject>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonParseError>
#include <QStringList>
#include <QDebug>

class JsonExplain : public QObject
{
    Q_OBJECT

public:
    explicit JsonExplain(QObject *parent = nullptr);
    void JsonGetCurrentState(const QString &messageReceived);

signals:
    void stateParsed(const QString &state);
    void jointAnglesParsed(const QList<int> &angles);
    void poseParsed(const QList<int> &pose);
    void errorsParsed(const QList<int> &errors);
};

#endif // JSONEXPLAIN_H
