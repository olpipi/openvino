#pragma once

#include <cstdio>
#include <limits>


namespace ov {
class ViewBuffer {
    class wrapbuf : public std::streambuf {
    public:
        wrapbuf(char* m_data, size_t m_size) {
            setg(m_data, m_data, m_data + m_size);
        }
    };


public:
    ViewBuffer(char* data, size_t size) : m_data(data), m_size(size), m_offset(0) {}

    char* get_next(size_t size) {
        if (size + m_offset > m_size)
            return nullptr;
        auto return_ptr = m_data + m_offset;
        m_offset += size;
        return return_ptr;
    }

    void getline (std::string& out_str) {
        wrapbuf buffer(m_data  + m_offset, std::min(static_cast<size_t>(std::numeric_limits<int32_t>::max()), m_size - m_offset));
        std::istream stream(&buffer);
        std::getline(stream, out_str);
        m_offset += out_str.size();
    }

private:
    char* m_data;
    size_t m_size;
    size_t m_offset;
};

}  //  namespace ov