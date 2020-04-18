#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include <string>

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


inline cl::Program
buildProgramFromFile(
  const std::string& baseName,
  const cl::Context& context,
  const std::vector<cl::Device>& devices,
  bool saveBinary = true)
{
  int cnt = 0;
  std::vector<std::vector<unsigned char>> loadedBinaries;
  do {
    std::ifstream ifs{baseName + "." + std::to_string(cnt) + ".bc", std::ios::binary};
    if (!ifs.is_open()) {
      cnt = 0;
      continue;
    }
    loadedBinaries.emplace_back(readBinaryAll(ifs));
  } while (cnt != 0);

  if (loadedBinaries.size() > 0) {
    cl::Program program(
      context,
      devices,
      loadedBinaries);
    program.build(devices);
    return program;
  }

  const auto sourceFileName = std::string{baseName + ".cl"};
  std::ifstream ifs{sourceFileName};
  if (!ifs.is_open()) {
    throw std::runtime_error{"Failed to open: " + sourceFileName};
  }
  cl::Program program = cl::Program(
    context,
    readTextAll(ifs));
  program.build(devices);

  if (!saveBinary) {
    return program;
  }

  const auto builtBinaries = program.getInfo<CL_PROGRAM_BINARIES>();
  for (const auto& binary : builtBinaries) {
    std::ofstream ofs{baseName + "." + std::to_string(cnt) + ".bc", std::ios::binary};
    if (!ofs.is_open()) {
      std::cerr << "Failed to open: " << "kernel.bc" << std::endl;
    }
    ofs.write(reinterpret_cast<const char*>(binary.data()), binary.size());
  }

  return program;
}

}  // namespace


int
main()
{
  constexpr auto kDataSize = 1000000;
  constexpr auto kEps = 1.0e-3f;
  const std::string sourceFileName{"kernel.cl"};

  cl_int err = CL_SUCCESS;
  try {
    std::cout << "Get platforms" << std::endl;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform not found" << std::endl;
      return -1;
    }

    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>((platforms[0])()),
      0
    };
    std::cout << "Create context" << std::endl;
    cl::Context context{CL_DEVICE_TYPE_GPU, properties};

    std::cout << "Get devices" << std::endl;
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    std::cout << "Build progam" << std::endl;
    auto program = buildProgramFromFile(
      "kernel",
      context,
      devices);

    std::cout << "Create kernel" << std::endl;
    cl::Kernel kernel{program, "innerProduct", &err};


    std::cout << "Create command queue" << std::endl;
    cl::Event event;
    cl::CommandQueue queue{context, devices[0], 0, &err};


    std::cout << "Allocate host/device buffer A" << std::endl;
    cl::Buffer deviceDataA{
      context,
      CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
      sizeof(float) * kDataSize};
    std::cout << "Retrieve pointer available on host from buffer A" << std::endl;
    const auto ptrA = static_cast<float*>(queue.enqueueMapBuffer(
      deviceDataA,
      CL_TRUE,
      CL_MAP_WRITE,
      0,
      sizeof(float) * kDataSize));

    std::cout << "Allocate host/device buffer B" << std::endl;
    cl::Buffer deviceDataB{
      context,
      CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
      sizeof(float) * kDataSize};
    std::cout << "Retrieve pointer available on host from buffer B" << std::endl;
    const auto ptrB = static_cast<float*>(queue.enqueueMapBuffer(
      deviceDataB,
      CL_TRUE,
      CL_MAP_WRITE,
      0,
      sizeof(float) * kDataSize));

    std::cout << "Initialize buffer A and B" << std::endl;
    for (int i = 0; i < kDataSize; i++) {
      ptrA[i] = static_cast<float>(i);
      ptrB[i] = static_cast<float>(kDataSize - i);
    }

    std::cout << "Release pointer from buffer A" << std::endl;
    queue.enqueueUnmapMemObject(deviceDataA, ptrA);
    std::cout << "Release pointer from buffer B" << std::endl;
    queue.enqueueUnmapMemObject(deviceDataB, ptrB);


    std::cout << "Allocate host buffer C1 for host calculation" << std::endl;
    std::vector<float> hostDataC1(kDataSize);  // for answer (host)

    std::cout << "Multiply calculation on host: ";
    const auto start1 = std::chrono::high_resolution_clock::now();
    for (decltype(hostDataC1)::size_type i = 0; i < hostDataC1.size(); i++) {
      hostDataC1[i] = ptrA[i] * ptrB[i];
    }
    const auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count();
    std::cout << elapsed1 << " ms" << std::endl;

    std::cout << "Allocate host/device buffer C" << std::endl;
    cl::Buffer deviceDataC{
      context,
      CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
      sizeof(decltype(hostDataC1)::value_type) * hostDataC1.size()};
    kernel.setArg(0, deviceDataC);
    kernel.setArg(1, deviceDataA);
    kernel.setArg(2, deviceDataB);

    std::cout << "Multiply calculation on device: ";
    const auto start2 = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(kDataSize, 1, 1),
      cl::NullRange,
      nullptr,
      &event);
    event.wait();
    const auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2).count();
    std::cout << elapsed2 << " ms" << std::endl;

    std::cout << "Retrieve pointer available on host from buffer C" << std::endl;
    const auto ptrC2 = static_cast<float*>(queue.enqueueMapBuffer(
      deviceDataC,
      CL_TRUE,
      CL_MAP_READ,
      0,
      sizeof(float) * kDataSize));


    std::cout << "Verify calculation results... ";
    const auto verifyResult = std::equal(
      std::cbegin(hostDataC1),
      std::cend(hostDataC1),
      ptrC2,
      [&kEps](const auto& x, const auto& y) {
        return std::abs(x - y) < kEps;
      });
    if (verifyResult) {
      std::cout << "OK" << std::endl;
    } else {
      std::cout << "NG" << std::endl;
    }

    std::cout << "Unsync device buffer C and host buffer C2" << std::endl;
    queue.enqueueUnmapMemObject(deviceDataC, ptrC2);
  } catch (const cl::Error& ex) {
    std::cerr << "ERROR: " << ex.what() << "(" << ex.err() << ")" << std::endl;
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
