#include "test_plugin_trt.h"

#include <vector>
#include <cassert>
#include <cstring>

namespace myrelu
{
  // Helper function for serializing plugin
  template <typename T>
  void writeToBuffer(char *&buffer, const T &val)
  {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
  }

  // Helper function for deserializing plugin
  template <typename T>
  T readFromBuffer(const char *&buffer)
  {
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
  }

  int gDataSize = 5000;
} // namespace myrelu

namespace nvinfer1
{
  PluginFieldCollection TestReluPluginCreator::mFC{};
  std::vector<PluginField> TestReluPluginCreator::mPluginAttributes;

  TestRelu::TestRelu(int height, int width, int channel)
      : _mHeigth(height), _mWidth(width), _mChannel(channel)
  {
    _msize = _mHeigth * _mWidth * _mChannel;
    std::cout << ">>>>> TestRelu Constructer : "
              << _mHeigth << " "
              << _mWidth << " "
              << _mChannel << " "
              << _msize << "  <<<<<"
              << std::endl;
    _mTest = myrelu::gDataSize;
    std::cerr << "TestRelu construct = " << _mTest << std::endl;
  }

  TestRelu::TestRelu(const void *data, size_t length)
  {
    std::cout << ">>>>> TestRelu deSerial data！ <<<<<" << std::endl;
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    _msize = myrelu::readFromBuffer<int>(d);
    _mWidth = myrelu::readFromBuffer<int>(d);
    _mHeigth = myrelu::readFromBuffer<int>(d);
    _mChannel = myrelu::readFromBuffer<int>(d);

    std::cerr << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;

    assert(d == a + length);
    // _msize = *reinterpret_cast<const size_t *>(buffer);
  }

  const char *TestRelu::getPluginType() const
  {
    return MYRELU_PLUGIN_NAME;
  }

  const char *TestRelu::getPluginVersion() const
  {
    return MYRELU_PLUGIN_VERSION;
  }

  int TestRelu::getNbOutputs() const
  {
    return 1;
  }

  Dims TestRelu::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
  {
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[index].nbDims == 3);

    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]); // C*H*W
  }

  bool TestRelu::supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const
  {
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
  }

  int TestRelu::initialize()
  {
    std::cout << ">>>>> TestRelu::initialize <<<<" << std::endl;
    return 0;
  }

  size_t TestRelu::getWorkspaceSize(int maxBatchSize) const
  {
    return 0;
  }

  size_t TestRelu::getSerializationSize() const
  {
    return sizeof(_msize) + sizeof(_mHeigth) + sizeof(_mWidth) + sizeof(_mChannel);
  }

  void TestRelu::serialize(void *buffer) const
  {
    std::cerr << ">>>> TestRelu::serialize <<<<" << std::endl;
    char *d = static_cast<char *>(buffer);
    const char *a = d;
    std::cout << ">>>>> TestRelu::serialize : "
              << _mHeigth << " "
              << _mWidth << " "
              << _mChannel << " "
              << _msize << "  <<<<<"
              << std::endl;

    myrelu::writeToBuffer(d, _msize);
    myrelu::writeToBuffer(d, _mHeigth);
    myrelu::writeToBuffer(d, _mWidth);
    myrelu::writeToBuffer(d, _mChannel);

    assert(d == a + getSerializationSize());

    // *reinterpret_cast<size_t *>(buffer) = _msize;
  }

  void TestRelu::destroy()
  {
    std::cerr << " >>>> TestRelu::destroy() <<<< " << std::endl;
    delete this;
  }

  IPluginV2IOExt *TestRelu::clone() const
  {
    std::cerr << ">>>> TestRelu::clone() <<<< " << std::endl;
    auto reluPlugin = new TestRelu(_mHeigth, _mWidth, _mChannel);
    reluPlugin->setPluginNamespace(_mNamespace.c_str());
    return reluPlugin;
  }

  void TestRelu::setPluginNamespace(const char *pluginNamespace)
  {
    _mNamespace = pluginNamespace;
  }

  const char *TestRelu::getPluginNamespace() const
  {
    return _mNamespace.c_str();
  }

  int TestRelu::enqueue(int batchSize, const void *const *inputs,
                        void **outputs, void *workspace, cudaStream_t stream)
  {
    // int block_size = 256;
    // int grid_size = (msize + block_size - 1) / block_size;

    // _leakyReluKer<<<grid_size, block_size>>>(
    //     reinterpret_cast<float const *>(inputs[0]),
    //     reinterpret_cast<float *>(outputs[0]), msize);

    std::cerr << "TestRelu::enqueue,_mTest === " << _mTest << std::endl;
    reluInference(stream, _msize, inputs, outputs);
    return 0;
  }

  DataType TestRelu::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  {
    return DataType::kFLOAT;
  }

  // Return true if output tensor is broadcast across a batch.
  bool TestRelu::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const
  {
    return false;
  }

  bool TestRelu::canBroadcastInputAcrossBatch(int inputIndex) const
  {
    return false;
  }

  void TestRelu::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator)
  {
  }

  void TestRelu::configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput)
  {
    std::cout << "configurePlugin ---> "
              << in->dims.d[0] << " "
              << in->dims.d[1] << " "
              << in->dims.d[2] << std::endl;

    myrelu::gDataSize = in->dims.d[0] * in->dims.d[1] * in->dims.d[2]; // 1(channels)*28(height)*28(width)
  }

  void TestRelu::detachFromContext()
  {
  }

  ///////////////// TODO to fill data to construct /////////////////////
  TestReluPluginCreator::TestReluPluginCreator()
  {
    std::cout << ">>>> TestReluPlugin Creator <<<<" << std::endl;
    mPluginAttributes.clear();

    const static int data_heigth = 28;
    const static int data_width = 28;
    const static int data_channel = 1;

    mPluginAttributes.emplace_back(PluginField("height", &data_heigth, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("width", &data_heigth, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("channel", &data_channel, PluginFieldType::kINT8, 1));

    // // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  const char *TestReluPluginCreator::getPluginName() const
  {
    return MYRELU_PLUGIN_NAME;
  }

  const char *TestReluPluginCreator::getPluginVersion() const
  {
    return MYRELU_PLUGIN_VERSION;
  }

  const char *TestReluPluginCreator::getPluginNamespace() const
  {
    return _mNamespace.c_str();
  }

  void TestReluPluginCreator::setPluginNamespace(const char *pluginNamespace)
  {
    _mNamespace = pluginNamespace;
  }

  const PluginFieldCollection *TestReluPluginCreator::getFieldNames()
  {
    return &mFC;
  }

  IPluginV2 *TestReluPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
  {
    int imgHeight, imgWidth, imgChannel;
    const PluginField *fields = fc->fields;

    // Parse fields from PluginFieldCollection
    assert(fc->nbFields == 3);

    for (int i = 0; i < fc->nbFields; i++)
    {
      if (strcmp(fields[i].name, "height") == 0)
      {
        assert(fields[i].type == PluginFieldType::kINT8);
        imgHeight = *(static_cast<const int *>(fields[i].data));
      }
      else if (strcmp(fields[i].name, "width") == 0)
      {
        assert(fields[i].type == PluginFieldType::kINT8);
        imgWidth = *(static_cast<const int *>(fields[i].data));
      }
      else if (strcmp(fields[i].name, "channel") == 0)
      {
        assert(fields[i].type == PluginFieldType::kINT8);
        imgChannel = *(static_cast<const int *>(fields[i].data));
      }
      // std::cout << "TestReluPluginCreator === " << fields[i].data << std::endl;
      // std::cout << "TestReluPluginCreator === " << *(static_cast<const int *>(fields[i].data)) << std::endl;
    }

    // TestRelu *obj = new TestRelu(imgHeight, imgWidth, imgChannel);
    // obj->setPluginNamespace(_mNamespace.c_str());
    // return obj;
    return new TestRelu(imgHeight, imgWidth, imgChannel);
  }

  IPluginV2 *TestReluPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
  {
    // return new TestRelu(name, serialData, serialLength);

    TestRelu *obj = new TestRelu(serialData, serialLength);
    obj->setPluginNamespace(_mNamespace.c_str());
    return obj;
  }

} // namespace nvinfer1
