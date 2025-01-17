#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <ctime>

enum LogLevel {
    INFO,
    WARNING,
    ERROR,
    FATAL
};

class Logger {
public:
    Logger() : log_level(INFO) {}

    void setLogLevel(LogLevel level) {
        log_level = level;
    }

    void log(LogLevel level, const std::string& file, int line, const std::string& message) {
        if (level >= log_level) {
            std::string color;
            switch (level) {
                case INFO:    color = "\033[32m"; break; // 绿色
                case WARNING: color = "\033[33m"; break; // 黄色
                case ERROR:   color = "\033[31m"; break; // 红色
                case FATAL:   color = "\033[35m"; break; // 紫色
                default:      color = "\033[0m";  break; // 默认
            }
            std::cout << color << getCurrentTime() << " [" << getLevelString(level) << "] "
                      << file << ":" << line << " " << message << "\033[0m" << std::endl;
        }
    }

private:
    LogLevel log_level;

    std::string getCurrentTime() {
        std::time_t now = std::time(nullptr);
        std::tm* localTime = std::localtime(&now);
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", localTime);
        return timeStr;
    }

    std::string getLevelString(LogLevel level) {
        switch (level) {
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            case FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
};

class LogStream {
public:
    LogStream(LogLevel level, const std::string& file, int line) 
        : level(level), file(file), line(line) {}

    ~LogStream() {
        if (!buffer.str().empty()) {
            logger.log(level, file, line, buffer.str());
        }
    }

    template<typename T>
    LogStream& operator<<(const T& value) {
        buffer << value;
        return *this;
    }

    // 特化处理 std::endl
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        manip(buffer);  // 应用流操作符
        return *this;
    }
    static Logger logger; // 声明静态成员
private:
    LogLevel level;
    std::string file;
    int line;
    std::ostringstream buffer;
};


#define LOG(level) LogStream(level, __FILE__, __LINE__)

#endif // LOGGER_H
