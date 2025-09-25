#ifndef TIMER
#define TIMER

#include <exception>
#include <string>
#include <sys/time.h>

class Timer {
public:
    class TimerException : public std::exception {
    public:
        TimerException(const char *msgStr) throw() : msg(msgStr) {}
        virtual ~TimerException() throw() {}
        virtual const char* what() const throw() { return msg.c_str(); }
    private:
        std::string msg;
    };

    Timer() {
        running = false;
        ready = false;
    }
    void begin() {
        gettimeofday(&begin_time, NULL);
        running = true;
        ready = false;
    }
    void end() {
        if(!running)
            throw TimerException("The timer is not running.");
        gettimeofday(&end_time, NULL);
        running = false;
        ready = true;
    }
    double getTime() const {
        return getEndTime() - getBeginTime();
    }
    double getBeginTime() const {
        if(!running && !ready)
            throw TimerException("The timer is not running or not ready.");
        return begin_time.tv_sec + begin_time.tv_usec*1.0E-6;
    }
    double getEndTime() const {
        if(!ready)
            throw TimerException("The timer is not ready.");
        return end_time.tv_sec + end_time.tv_usec*1.0E-6;
    }
private:
    bool running ;
    bool ready;
    struct timeval begin_time;
    struct timeval end_time;
};

#endif