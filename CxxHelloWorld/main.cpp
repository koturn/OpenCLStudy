#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
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
  const std::string sourceFileName{"kernel.cl"};

  cl_int err = CL_SUCCESS;
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      std::cerr << "Platform size 0" << std::endl;
      return -1;
    }

    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>((platforms[0])()),
      0
    };
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    auto program = buildProgramFromFile(
      "kernel",
      context,
      devices);

    cl::CommandQueue queue{context, devices[0], 0, &err};

    auto kernelFunc = cl::KernelFunctor<>{program, "hello"};
    kernelFunc(
      cl::EnqueueArgs{
        queue,
        cl::NullRange,
        cl::NDRange(4, 4),
        cl::NullRange}).wait();
  } catch (const cl::Error& ex) {
    std::cerr << "ERROR: " << ex.what() << "(" << ex.err() << ")" << std::endl;
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }
}
