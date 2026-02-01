#include "micromanipulator.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Micromanipulator w;
    w.show();

    return a.exec();
}
