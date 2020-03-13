#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

#include <opencl.hpp>


int
main()
{
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "No platform found.\n";
      return 1;
    }
    int i = 0;
    for (const auto& platform : platforms) {
      std::cout << "Platform #" << i << ":\n";
      std::cout << "  Profile: " << platform.getInfo<CL_PLATFORM_PROFILE>() << "\n";
      std::cout << "  Name: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
      std::cout << "  Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
      std::cout << "  Extensions: " << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      int j = 0;
      for (const auto& device : devices) {
        {
          cl_device_type deviceType = device.getInfo<CL_DEVICE_TYPE>();
          std::string deviceTypeStr = deviceType == CL_DEVICE_TYPE_CPU ? "CPU"
            : deviceType == CL_DEVICE_TYPE_GPU ? "GPU"
            : deviceType == CL_DEVICE_TYPE_ACCELERATOR ? "Accelerator"
            : "unknown";
          std::cout << "  Device #" << j << " (" << deviceTypeStr << "):\n";
        }
        std::cout << "    Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << "    Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
        std::cout << "    Device Version: " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
        std::cout << "    Driver Version: " << device.getInfo<CL_DRIVER_VERSION>() << "\n";
        std::cout << "    Extensions: " << device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";
        std::cout << "    Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
        std::cout << "    Preferred Vector Width (Float): " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() << "\n";
        std::cout << "    Preferred Vector Width (Double): " << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>() << "\n";
        ++j;
      }
      ++i;
    }
    std::cout << std::flush;
    return 0;
  } catch (cl::Error const& ex) {
    std::cerr << "OpenCL Error: " << ex.what() << " (code " << ex.err() << ")" << std::endl;
    return 1;
  } catch (std::exception const& ex) {
    std::cerr << "Exception: " << ex.what() << std::endl;
    return 1;
  }
}
