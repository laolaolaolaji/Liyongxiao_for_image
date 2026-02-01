QT       += core gui
QT       += multimedia
QT       += serialport
QT       += network
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    camera_module.cpp \
    image_processor.cpp \
    jsonexplain.cpp \
    macro_arm.cpp \
    macro_micro_controller.cpp \
    main.cpp \
    micromanipulator.cpp \
    micro_arm.cpp

HEADERS += \
    camera_module.h \
    image_processor.h \
    jsonexplain.h \
    macro_arm.h \
    macro_micro_controller.h \
    micromanipulator.h \
    micro_arm.h

FORMS += \
    micromanipulator.ui

INCLUDEPATH += $$PWD/include
win32 {
    exists($$PWD/lib/api_cpp.lib) {
        LIBS += -L$$PWD/lib -lapi_cpp
    } else {
        message("api_cpp.lib not found in $$PWD/lib; skipping linkage")
    }
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

LIBS += -LD:/Opencv/opencv/build/x64/vc16/lib/ -lopencv_world481d
INCLUDEPATH += D:/Opencv/opencv/build/include
DEPENDPATH += D:/Opencv/opencv/build/x64/vc16/include

DISTFILES += \
    text01

LIBS += -luser32
