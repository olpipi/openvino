// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Cache Manager class C++ API
 *
 * @file cache_manager.hpp
 */
#pragma once

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <windows.h>

#include "openvino/util/file_util.hpp"

namespace ov {

/**
 * @brief This class limits the locale env to a special value in sub-scope
 *
 */
class ScopedLocale {
public:
    ScopedLocale(int category, std::string newLocale) : m_category(category) {
        m_oldLocale = setlocale(category, nullptr);
        setlocale(m_category, newLocale.c_str());
    }
    ~ScopedLocale() {
        setlocale(m_category, m_oldLocale.c_str());
    }

private:
    int m_category;
    std::string m_oldLocale;
};

/**
 * @brief This class represents private interface for Cache Manager
 *
 */
class ICacheManager {
public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
     *
     */
    using StreamWriter = std::function<void(std::ostream&)>;
    /**
     * @brief Callback when OpenVINO intends to write model to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, model will not be cached
     *
     * @param id Id of cache (hash of the model)
     * @param writer Lambda function to be called when stream is created
     */
    virtual void write_cache_entry(const std::string& id, StreamWriter writer) = 0;

    /**
     * @brief Function passing created input stream
     */
    using StreamReader = std::function<void(char*, size_t)>;

    /**
     * @brief Callback when OpenVINO intends to read model from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, model will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the model)
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void read_cache_entry(const std::string& id, StreamReader reader) = 0;

    /**
     * @brief Callback when OpenVINO intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the model)
     */
    virtual void remove_cache_entry(const std::string& id) = 0;
};

/**
 * @brief File storage-based Implementation of ICacheManager
 *
 * Uses simple file for read/write cached models.
 *
 */
class FileStorageCacheManager final : public ICacheManager {


    class istreambuf_view : public std::streambuf {
    public:
        istreambuf_view(std::vector<char>& vec)
            :  // ptr + size
              begin_(vec.data()),
              end_(vec.data() + vec.size()),
              current_(vec.data()) {}


    protected:
        int_type underflow() override {
            return (current_ == end_ ? traits_type::eof() : traits_type::to_int_type(*current_));
        }

        int_type uflow() override {
            return (current_ == end_ ? traits_type::eof() : traits_type::to_int_type(*current_++));
        }

        int_type pbackfail(int_type ch) override {
            if (current_ == begin_ || (ch != traits_type::eof() && ch != current_[-1]))
                return traits_type::eof();

            return traits_type::to_int_type(*--current_);
        }

        std::streamsize showmanyc() override {
            return end_ - current_;
        }

        const char* const begin_;
        const char* const end_;
        const char* current_;
    };


    template <typename CharT, typename TraitsT = std::char_traits<CharT>>
    class vectorwrapbuf : public std::basic_streambuf<CharT, TraitsT> {
    public:
        vectorwrapbuf(std::vector<CharT>& vec) {
            setg(vec.data(), vec.data(), vec.data() + vec.size());
        }
    };

    std::string m_cachePath;
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
    std::wstring getBlobFile(const std::string& blobHash) const {
        return ov::util::string_to_wstring(ov::util::make_path(m_cachePath, blobHash + ".blob"));
    }
#else
    std::string getBlobFile(const std::string& blobHash) const {
        return ov::util::make_path(m_cachePath, blobHash + ".blob");
    }
#endif

public:
    /**
     * @brief Constructor
     *
     */
    FileStorageCacheManager(std::string cachePath) : m_cachePath(std::move(cachePath)) {}

    /**
     * @brief Destructor
     *
     */
    ~FileStorageCacheManager() override = default;

private:
    void write_cache_entry(const std::string& id, StreamWriter writer) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        std::ofstream stream(getBlobFile(id), std::ios_base::binary | std::ofstream::out);
        writer(stream);
    }

    void read_cache_entry(const std::string& id, StreamReader reader) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        auto blobFileName = getBlobFile(id);
        if (ov::util::file_exists(blobFileName)) {
        std::chrono::steady_clock::time_point begin, end;
        begin = std::chrono::steady_clock::now();
#ifdef _WIN32

#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
            auto h = ::CreateFileW(blobFileName.c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   0,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL,
                                   0);
#    else
            auto h = ::CreateFileA(blobFileName.c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   0,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL,
                                   0);
#    endif

            LARGE_INTEGER file_size_large;
            if (::GetFileSizeEx(h, &file_size_large) == 0) {
                throw std::runtime_error("Can not get file size for ");
            }

            auto size = static_cast<uint64_t>(file_size_large.QuadPart);
            std::vector<char> data(size);


            size_t bytes_read = 0;
            while (bytes_read < size) {
                DWORD chunk_size = static_cast<DWORD>(std::min<size_t>(size - bytes_read, 64 * 1024 * 1024));
                DWORD chunk_read = 0;
                BOOL result = ReadFile(h, data.data() + bytes_read, chunk_size, &chunk_read, NULL);
                if (!result) {
                    auto error = GetLastError();
                    throw std::runtime_error("read error: %s" + GetLastError());
                    // throw std::runtime_error("blalba");
                }
                if (chunk_read < chunk_size || chunk_read == 0) {
                    throw std::runtime_error("unexpectedly reached end of file");
                }

                bytes_read += chunk_read;
            }
            ::CloseHandle(h);

            vectorwrapbuf<char> databuf(data);
            istreambuf_view databuf2(data);
            std::istream stream(&databuf);
#else
            std::ifstream stream(blobFileName, std::ios_base::binary);
#endif
            //end = std::chrono::steady_clock::now();
            //std::cout << "time to read file:"
            //          << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
            reader(stream);
            //end = std::chrono::steady_clock::now();
            //std::cout << "time to read + deserialize file:"
            //          << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";

        }
    }

    void remove_cache_entry(const std::string& id) override {
        auto blobFileName = getBlobFile(id);

        if (ov::util::file_exists(blobFileName)) {
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
            _wremove(blobFileName.c_str());
#else
            std::remove(blobFileName.c_str());
#endif
        }
    }
};

}  // namespace ov
