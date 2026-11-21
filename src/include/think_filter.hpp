/// \file think_filter.hpp
/// \brief Filter for reasoning / think blocks in interactive CLI streaming
#pragma once

#include <iostream>
#include <streambuf>
#include <vector>
#include <string>

class think_streambuf : public std::streambuf {
private:
    bool in_think_mode;
    bool has_started;
    std::string match_buf;
    std::ostream& out_stream;
    const std::string think_end_tag = "</think>";

public:
    think_streambuf(std::ostream& os, bool start_in_think = true)
        : out_stream(os), in_think_mode(start_in_think), has_started(false) {}

    ~think_streambuf() override {
        flush_buffer();
    }

protected:
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            char c = static_cast<char>(ch);
            process_char(c);
        }
        return ch;
    }

    int sync() override {
        flush_buffer();
        out_stream.flush();
        return 0;
    }

private:
    void process_char(char c) {
        if (!has_started) {
            has_started = true;
            if (in_think_mode) {
                out_stream << "\033[90m<think>\n";
            }
        }

        if (in_think_mode) {
            match_buf += c;
            if (think_end_tag.rfind(match_buf, 0) == 0) {
                if (match_buf == think_end_tag) {
                    out_stream << "</think>\033[0m\n\n";
                    match_buf.clear();
                    in_think_mode = false;
                }
            } else {
                size_t matched_len = 0;
                for (size_t len = match_buf.length() - 1; len > 0; len--) {
                    std::string suffix = match_buf.substr(match_buf.length() - len);
                    if (think_end_tag.rfind(suffix, 0) == 0) {
                        matched_len = len;
                        break;
                    }
                }
                out_stream << match_buf.substr(0, match_buf.length() - matched_len);
                match_buf = match_buf.substr(match_buf.length() - matched_len);
            }
        } else {
            out_stream.put(c);
        }
    }

    void flush_buffer() {
        if (!match_buf.empty()) {
            out_stream << match_buf;
            match_buf.clear();
        }
        if (in_think_mode) {
            out_stream << "\033[0m";
            in_think_mode = false;
        }
    }
};

class cli_think_filter : public std::ostream {
private:
    think_streambuf buf;
public:
    cli_think_filter(std::ostream& os, bool start_in_think = true)
        : std::ostream(&buf), buf(os, start_in_think) {}
};
