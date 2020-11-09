#ifndef _TEST_PLUGIN_TRT
#define _TEST_PLUGIN_TRT

#include "NvInfer.h"
#include "leak_relu.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

static const char *MYRELU_PLUGIN_VERSION{"1.0"};
static const char *MYRELU_PLUGIN_NAME{"TestRelu"};

namespace nvinfer1
{
  /**
   * @brief leakRelu plugin
   * 
   */
  class TestRelu : public IPluginV2IOExt
  {
  public:
    TestRelu(int height, int width, int channel);

    // 反序列化相关构造函数
    TestRelu(const void *data, size_t length);

    virtual ~TestRelu()
    {
      std::cerr << ">>>>> TestReLU released！ <<<<<" << std::endl;
    }

    // 返回输出的数量
    int getNbOutputs() const override;

    // 返回输出的维度（Dims数据结构）
    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

    // 为类的私有成员变量赋值？ 空
    int initialize() override;

    // 空
    void terminate() override {}

    // https://blog.csdn.net/cpongo3/article/details/93624265
    size_t getWorkspaceSize(int maxBatchSize) const override;

    // 核心函数
    int enqueue(int batchSize, const void *const *inputs,
                void **outputs, void *workspace, cudaStream_t stream) override;

    // 获取序列化长度
    size_t getSerializationSize() const override;

    // 用来匹配plugin creator返回的plugin name的方法，手动调用
    const char *getPluginType() const override;

    // 用来匹配plugin creator返回的plugin version的方法
    const char *getPluginVersion() const override;

    // 序列化
    void serialize(void *buffer) const override;

    // 定义datatype和format
    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const override;

    // 删除当前obj
    void destroy() override;

    // 每次创建包含此plugin的新builder，network或engine时，都会调用此方法
    IPluginV2IOExt *clone() const override;

    // 用来设置这个plugin对象属于哪个namesapce的方法，非必须
    void setPluginNamespace(const char *pluginNamespace) override;

    // 用来返回该plugin对象所属namespace的方法，非必须
    const char *getPluginNamespace() const override;

    // 获取输出数据格式
    DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) override;

    void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput) override;

    void detachFromContext() override;

  private:
    int _msize;
    int _mHeigth;
    int _mWidth;
    int _mChannel;

    int _mTest;
    std::string _mNamespace;
  };

  /**
   * @brief leakyRelu Plugin Creator
   * 
   */
  class TestReluPluginCreator : public IPluginCreator
  {
  public:
    TestReluPluginCreator();

    ~TestReluPluginCreator() override = default;

    const char *getPluginName() const override;

    const char *getPluginVersion() const override;

    const char *getPluginNamespace() const override;

    void setPluginNamespace(const char *pluginNamespace) override;

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override;

    const PluginFieldCollection *getFieldNames() override;

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

  private:
    std::string _mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
  };

} // namespace nvinfer1
#endif
