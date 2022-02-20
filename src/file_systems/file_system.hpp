#ifndef _FILE_SYSTEM_HPP
#define _FILE_SYSTEM_HPP

#include <string>

#include "../common/frame.hpp"

class FileSystem
{
    public:
        FileSystem();

        ~FileSystem();

        void SaveFrame(const Frame& frame);
        void ClearCacheFolder();

    private:
        std::string _cacheFolderPath;
}l

#endif // _FILE_SYSTEM_HPP