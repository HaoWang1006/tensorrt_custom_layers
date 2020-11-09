#include <vector>
#include <iostream>

using namespace std;

struct MyTest
{
  int width;
  int height;
  float a[10];
};

int main()
{
  std::vector<int> vec;
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  std::cout << "vec.data" << vec.data() << std::endl;

  std::cout << "sizeof(1*28*28)=== " << sizeof(1 * 28 * 28) << std::endl;

  int a = 1 * 28 * 28;
  int *buff = &a;
  std::cout << "^^^^^^^^^^^^^^^ === " << buff << std::endl;

  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int size = sizeof(buffer) / sizeof(buffer[0]);
  std::cout << "sizeof(buffer) === " << sizeof(buffer) << std::endl;
  std::cout << "sizeof(buffer[0]) === " << sizeof(buffer[0]) << std::endl;

  int var;
  int *ptr;
  int **pptr;

  var = 3000;

  // 获取 var 的地址
  ptr = &var;

  // 使用运算符 & 获取 ptr 的地址
  pptr = &ptr;

  // 使用 pptr 获取值
  cout << "var 值为 :" << var << endl;
  cout << "*ptr 值为:" << *ptr << endl;
  cout << "*pptr 值为:" << *pptr << endl;
  cout << "**pptr 值为:" << **pptr << endl;

  size_t abc = 10;
  int d = abc;
  std::cout << "size_t test === " << d << std::endl;

  std::cerr << sizeof(MyTest) << " float == "
            << sizeof(float) << " int === " << sizeof(int) << std::endl;

  return 0;
}