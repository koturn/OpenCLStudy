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

#if __cplusplus >= 201703L || defined(_MSVC_LANG) && _MSVC_LANG >= 201703L
#  include <cstdlib>
#elif defined(_MSC_VER) || defined(__MINGW32__)
#  include <malloc.h>
#else
#  include <cstdlib>
#endif  // defined(_MSC_VER) || defined(__MINGW32__)

#include <config/opencl.hpp>


namespace
{

template<typename T = void>
inline T*
alignedMalloc(std::size_t nBytes, std::size_t alignment = alignof(T)) noexcept
{
#if __cplusplus >= 201703L || defined(_MSVC_LANG) && _MSVC_LANG >= 201703L
  return reinterpret_cast<T*>(std::aligned_alloc(alignment, nBytes));
#elif defined(_MSC_VER) || defined(__MINGW32__)
  return reinterpret_cast<T*>(::_aligned_malloc(nBytes, alignment));
#else
  void* p;
  return reinterpret_cast<T*>(::posix_memalign(&p, alignment, nBytes) == 0 ? p : nullptr);
#endif  // defined(_MSC_VER) || defined(__MINGW32__)
}


template<typename T>
inline T*
alignedAllocArray(std::size_t size, std::size_t alignment = alignof(T)) noexcept
{
  return alignedMalloc<T>(size * sizeof(T), alignment);
}


inline void
alignedFree(void* ptr) noexcept
{
#if __cplusplus >= 201703L || defined(_MSVC_LANG) && _MSVC_LANG >= 201703L
  return std::free(ptr);
#elif defined(_MSC_VER) || defined(__MINGW32__)
  ::_aligned_free(ptr);
#else
  std::free(ptr);
#endif  // defined(_MSC_VER) || defined(__MINGW32__)
}


template<
  typename T,
  std::size_t kAlignment = alignof(T)
>
class AlignedAllocator
{
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = typename std::add_pointer<value_type>::type;
  using const_pointer = typename std::add_pointer<const value_type>::type;

  template<class U>
  struct rebind
  {
    using other = AlignedAllocator<U, kAlignment>;
  };

  AlignedAllocator() noexcept
  {}

  template<typename U>
  AlignedAllocator(const AlignedAllocator<U, kAlignment>&) noexcept
  {}

  pointer
  allocate(size_type n, const_pointer /* hint */ = nullptr) const
  {
    auto p = alignedAllocArray<value_type>(n, kAlignment);
    if (p == nullptr) {
      throw std::bad_alloc{};
    }
    return p;
  }

  void
  deallocate(pointer p, size_type /* n */) const noexcept
  {
    alignedFree(p);
  }
};  // class AlignedAllocator


template<
  typename T,
  std::size_t kAlignment1,
  typename U,
  std::size_t kAlignment2
>
inline bool
operator==(const AlignedAllocator<T, kAlignment1>&, const AlignedAllocator<U, kAlignment2>&) noexcept
{
  return kAlignment1 == kAlignment2;
}


template<
  typename T,
  std::size_t kAlignment1,
  typename U,
  std::size_t kAlignment2
>
inline bool
operator!=(const AlignedAllocator<T, kAlignment1>& lhs, const AlignedAllocator<U, kAlignment2>& rhs) noexcept
{
  return !(lhs == rhs);
}


template<
  typename T,
  typename U
>
inline constexpr T
calcPotAlignedSize(T x, U n) noexcept
{
  static_assert(std::is_integral<T>::value, "[calcPotAlignedSize] Type of first argument must be integral.");
  static_assert(std::is_integral<U>::value, "[calcPotAlignedSize] Type of second argument must be integral.");
  const auto m = (1 << n) - 1;
  return (x + m) & ~m;
}


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
  constexpr auto kAlignment = calcPotAlignedSize(1, 12);
  constexpr auto kDataSize = calcPotAlignedSize(1000000, 6);
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


    std::cout << "Allocate host buffer A" << std::endl;
    std::vector<float, AlignedAllocator<float, kAlignment>> hostDataA(kDataSize);
    std::cout << "Allocate host buffer B" << std::endl;
    std::vector<float, AlignedAllocator<float, kAlignment>> hostDataB(kDataSize);
    for (decltype(hostDataA)::size_type i = 0; i < hostDataA.size(); i++) {
      hostDataA[i] = static_cast<float>(i);
      hostDataB[i] = static_cast<float>(hostDataA.size() - i);
    }
    std::cout << "Allocate host buffer C1" << std::endl;
    std::vector<float, AlignedAllocator<float, kAlignment>> hostDataC1(kDataSize);  // for answer (host)

    std::cout << "Multiply calculation on host: ";
    const auto start1 = std::chrono::high_resolution_clock::now();
    for (decltype(hostDataC1)::size_type i = 0; i < hostDataC1.size(); i++) {
      hostDataC1[i] = hostDataA[i] * hostDataB[i];
    }
    const auto elapsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count();
    std::cout << elapsed1 << " ms" << std::endl;

    std::cout << "Allocate host buffer C2" << std::endl;
    std::vector<float, AlignedAllocator<float, kAlignment>> hostDataC2(kDataSize);  // for answer (device)
    std::cout << "Bind host buffer C2 to device buffer C" << std::endl;
    cl::Buffer deviceDataC{
      context,
      CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(decltype(hostDataC2)::value_type) * hostDataC2.size(),
      hostDataC2.data()};
    kernel.setArg(0, deviceDataC);

    std::cout << "Bind host buffer A to device buffer A" << std::endl;
    cl::Buffer deviceDataA{
      context,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(decltype(hostDataA)::value_type) * hostDataA.size(),
      hostDataA.data()};
    kernel.setArg(1, deviceDataA);

    std::cout << "Bind host buffer B to device buffer B" << std::endl;
    cl::Buffer deviceDataB{
      context,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(decltype(hostDataB)::value_type) * hostDataB.size(),
      hostDataB.data()};
    kernel.setArg(2, deviceDataB);

    std::cout << "Create command queue" << std::endl;
    cl::Event event;
    cl::CommandQueue queue{context, devices[0], 0, &err};

    std::cout << "Multiply calculation on device: ";
    const auto start2 = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(hostDataA.size(), 1, 1),
      cl::NullRange,
      nullptr,
      &event);
    event.wait();
    const auto elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2).count();
    std::cout << elapsed2 << " ms" << std::endl;

    std::cout << "Synchronize device buffer C and host buffer C2" << std::endl;
    const auto ptrC2 = queue.enqueueMapBuffer(
      deviceDataC,
      CL_TRUE,
      CL_MAP_READ,
      0,
      sizeof(decltype(hostDataC2)::value_type) * hostDataC2.size());

    std::cout << "ptrC2 " << ((ptrC2 == hostDataC2.data()) ? "==" : "!=") << " hostDataC2.data()" << std::endl;

    std::cout << "Verify calculation results... ";
    const auto verifyResult = std::equal(
      std::cbegin(hostDataC1),
      std::cend(hostDataC1),
      std::cbegin(hostDataC2),
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
