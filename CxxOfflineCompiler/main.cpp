#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <ArgumentParser.hpp>
#include <config/opencl.hpp>


namespace
{

inline std::string
removeSuffix(const std::string& filename) noexcept
{
  return filename.substr(0, filename.find_last_of("."));
}


inline std::size_t
getStreamSize(std::istream& is) noexcept
{
  const auto currentPos = is.tellg();
  is.seekg(0, std::ifstream::end);
  const auto endPos = is.tellg();
  is.seekg(currentPos, std::ifstream::beg);
  return static_cast<std::size_t>(endPos - currentPos);
}


inline std::vector<unsigned char>
readBinaryAll(std::ifstream& ifs) noexcept
{
  std::vector<unsigned char> binary(getStreamSize(ifs));
  ifs.read(reinterpret_cast<char*>(&binary[0]), binary.size());
  return binary;
}


inline std::string
readTextAll(std::ifstream& ifs) noexcept
{
  std::string text;
  text.resize(getStreamSize(ifs));
  ifs.read(&text[0], text.size());
  return text;
}


inline void
showPlatformInfo(const cl::Platform& platform, cl_int deviceType) noexcept
{
  std::cout << "  Profile: " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
  std::cout << "  Name: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
  std::cout << "  Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
  std::cout << "  Extensions: " << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
  std::vector<cl::Device> devices;
  platform.getDevices(deviceType, &devices);
  int deviceIndex = 0;
  for (const auto& device : devices) {
    {
      const auto dt = device.getInfo<CL_DEVICE_TYPE>();
      std::string deviceTypeStr = dt == CL_DEVICE_TYPE_CPU ? "CPU"
        : dt == CL_DEVICE_TYPE_GPU ? "GPU"
        : dt == CL_DEVICE_TYPE_ACCELERATOR ? "Accelerator"
        : "unknown";
      std::cout << "  Device #" << deviceIndex << " (" << deviceTypeStr << "):\n";
    }
    std::cout << "    Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "    Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
    std::cout << "    Device Version: " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
    std::cout << "    Driver Version: " << device.getInfo<CL_DRIVER_VERSION>() << "\n";
    std::cout << "    Extensions: " << device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";
    std::cout << "    Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
    std::cout << "    Preferred Vector Width (Float): " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() << "\n";
    std::cout << "    Preferred Vector Width (Double): " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>() << "\n";
    deviceIndex++;
  }
  std::cout << std::flush;
}


inline void
showPlatformInfo(const std::vector<cl::Platform>& platforms, cl_int deviceType) noexcept
{
  int platformIndex = 0;
  for (const auto& platform : platforms) {
    std::cout << "Platform #" << platformIndex << ":\n";
    showPlatformInfo(platform, deviceType);
    platformIndex++;
  }
}


}  // namespace


int
main(int argc, const char* argv[])
{
  std::unordered_map<std::string, cl_int> kDeviceTypeMap{
    {"all", CL_DEVICE_TYPE_ALL},
    {"default", CL_DEVICE_TYPE_DEFAULT},
    {"cpu", CL_DEVICE_TYPE_CPU},
    {"gpu", CL_DEVICE_TYPE_GPU},
    {"accelerator", CL_DEVICE_TYPE_ACCELERATOR}
  };

  cl_int err = CL_SUCCESS;
  try {
    ArgumentParser ap{argv[0]};
    // ap.add("all", 'a', ArgumentParser::OptionType::kNoArgument, false, "Compile kernel program for all detected devices");
    ap.add('l', "list", ArgumentParser::OptionType::kNoArgument, "List up all platforms and devices");
    ap.add('t', "device-type", ArgumentParser::OptionType::kRequiredArgument,
        "Specify device type\n"
        "      all: CPU and GPU\n"
        "      cpu: CPU only\n"
        "      gpu: GPU only\n"
        "      accelerator: Accelerator only", "DEVICE_TYPE", "default");
    // ap.add('o', "output", ArgumentParser::OptionType::kRequiredArgument, "", "Specify output file name", "FILE_NAME");
    ap.add('O', "option", ArgumentParser::OptionType::kRequiredArgument, "Specify compile option", "COMPILE_OPTION",
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
        "-cl-std=CL2.0"
#else
        ""
#endif // #if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
    );
    // ap.add('p', "platform", ArgumentParser::OptionType::kRequiredArgument, 0, "Specify platform index", "PLATFORM_INDEX");
    // ap.add('d', "device", ArgumentParser::OptionType::kRequiredArgument, 0, "Specify device index", "DEVICE_INDEX");
    ap.add("fsyntax-only", ArgumentParser::OptionType::kNoArgument, "Check syntax only, not generate binary");
    ap.add('h', "help", "Show help and exit this program");
    ap.parse(argc, argv);

    if (ap.get<bool>("help")) {
      ap.showUsage();
      return 0;
    }

    std::cout << "Find platforms ... ";
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform not found" << std::endl;
      return -1;
    }
    std::cout << platforms.size() << " platforms" << std::endl;

    const auto apDeviceType = ap.get("device-type");
    const auto deviceType = (kDeviceTypeMap.find(apDeviceType) == std::end(kDeviceTypeMap)) ? CL_DEVICE_TYPE_ALL : kDeviceTypeMap[apDeviceType];
    if (ap.get<bool>("list")) {
      showPlatformInfo(platforms, deviceType);
      return 0;
    }


    auto args = ap.getArguments();
    if (args.size() < 1) {
      std::cerr << "Please specify only one or more source file" << std::endl;
      return EXIT_FAILURE;
    }
    const auto isSaveBinary = !ap.get<bool>("fsyntax-only");
    const auto compileOption = ap.get("option");

    int platformIndex = 0;
    for (auto&& platform : platforms) {
      std::cout << "Platform " << platformIndex << std::endl;
      cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform()),
        0
      };
      cl::Context context{deviceType, properties};

      std::cout << "Find devices ... ";
      std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
      std::cout << devices.size() << " devices" << std::endl;

      std::cout << "Start building programs" << std::endl;

      for (const auto& sourceFileName : args) {
        std::cout << "Start compiling " << sourceFileName << std::endl;
        std::ifstream ifs{sourceFileName};
        if (!ifs.is_open()) {
          throw std::runtime_error{"Failed to open: " + sourceFileName};
        }
        cl::Program program = cl::Program{
          context,
          readTextAll(ifs)};
        try {
          program.build(devices, compileOption.c_str());
        } catch (const cl::Error&) {
          cl_int buildErr = CL_SUCCESS;
          auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
          for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl;
          }
        }

        if (isSaveBinary) {
          int cnt = 0;
          const auto baseName = removeSuffix(sourceFileName);
          const auto builtBinaries = program.getInfo<CL_PROGRAM_BINARIES>();
          for (const auto& binary : builtBinaries) {
            const std::string outputFileName{baseName + "." + std::to_string(platformIndex) + "." + std::to_string(cnt) + ".bc"};
            std::ofstream ofs{outputFileName, std::ios::binary};
            if (!ofs.is_open()) {
              throw std::runtime_error{"Failed to open: " + outputFileName};
            }
            ofs.write(reinterpret_cast<const char*>(binary.data()), binary.size());
          }
        }
      }
      std::cout << "finished building programs" << std::endl;

      if (!ap.get<bool>("all")) {
        break;
      }
    }
  } catch (const cl::Error& ex) {
    std::cerr << "ERROR: " << ex.what() << "(" << ex.err() << ")" << std::endl;
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
