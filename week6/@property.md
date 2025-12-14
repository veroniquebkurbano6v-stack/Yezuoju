# Python @property 装饰器详解

## 目录
1. [什么是@property装饰器](#什么是property装饰器)
2. [为什么需要@property](#为什么需要property)
3. [基本用法](#基本用法)
4. [进阶用法](#进阶用法)
5. [实际应用场景](#实际应用场景)
6. [常见问题与最佳实践](#常见问题与最佳实践)
7. [总结](#总结)

## 什么是@property装饰器

@property是Python中的一个内置装饰器，它允许我们将类的方法当作属性来访问。这意味着我们可以在不改变类外部调用方式的情况下，为类的属性添加额外的逻辑处理。

@property装饰器将一个方法转换为属性访问，使得我们可以像访问普通属性一样调用这个方法，而不需要使用括号。

## 为什么需要@property

在面向对象编程中，我们经常需要控制对类属性的访问。传统的方法是使用getter和setter方法，但这会导致代码冗长且不够Pythonic。

@property装饰器提供了一种更优雅的方式来控制属性的访问和修改，同时保持了代码的简洁性和可读性。

## 基本用法

### 只读属性

```python
class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

# 使用示例
person = Person("张", "三")
print(person.full_name)  # 输出: 张 三
# 注意：这里不需要加括号，因为full_name现在是属性而不是方法
```

### 可读写属性

```python
class Temperature:
    def __init__(self):
        self._celsius = 0

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度(-273.15°C)")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

# 使用示例
temp = Temperature()
temp.celsius = 25  # 调用setter方法
print(temp.celsius)  # 输出: 25，调用getter方法
print(temp.fahrenheit)  # 输出: 77.0

temp.fahrenheit = 86  # 调用fahrenheit的setter方法
print(temp.celsius)  # 输出: 30.0
```

### 只删除属性

```python
class Student:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.deleter
    def name(self):
        print("删除姓名属性")
        del self._name

# 使用示例
student = Student("李四")
print(student.name)  # 输出: 李四
del student.name  # 输出: 删除姓名属性
# print(student.name)  # 现在会抛出AttributeError，因为_name已被删除
```

## 进阶用法

### 使用@property进行验证

```python
class Product:
    def __init__(self, price):
        self.price = price

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("价格不能为负数")
        self._price = value

# 使用示例
product = Product(100)
product.price = 200  # 正常
# product.price = -50  # 抛出ValueError: 价格不能为负数
```

### 计算属性

```python
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius

# 使用示例
circle = Circle(5)
print(f"面积: {circle.area:.2f}")  # 输出: 面积: 78.54
print(f"周长: {circle.perimeter:.2f}")  # 输出: 周长: 31.42
```

### 缓存属性

```python
class ExpensiveComputation:
    def __init__(self):
        self._result = None

    @property
    def result(self):
        if self._result is None:
            print("执行复杂计算...")
            self._result = sum(range(1000000))  # 模拟耗时计算
        return self._result

# 使用示例
expensive = ExpensiveComputation()
print(expensive.result)  # 第一次访问会执行计算
print(expensive.result)  # 第二次访问直接返回缓存结果
```

## 实际应用场景

### 1. 数据验证

```python
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, value):
        if '@' not in value:
            raise ValueError("无效的电子邮件地址")
        self._email = value

# 使用示例
user = User("张三", "zhangsan@example.com")
user.email = "new@example.com"  # 正常
# user.email = "invalid-email"  # 抛出ValueError
```

### 2. API响应封装

```python
class APIResponse:
    def __init__(self, data):
        self._data = data

    @property
    def is_success(self):
        return self._data.get('status') == 'success'

    @property
    def error_message(self):
        return self._data.get('error', '未知错误')

    @property
    def content(self):
        return self._data.get('data', {})

# 使用示例
response = APIResponse({"status": "success", "data": {"id": 1, "name": "测试"}})
if response.is_success:
    print(response.content["name"])  # 输出: 测试
else:
    print(response.error_message)
```

### 3. 配置管理

```python
class Config:
    def __init__(self):
        self._settings = {}

    @property
    def debug_mode(self):
        return self._settings.get('debug', False)

    @debug_mode.setter
    def debug_mode(self, value):
        self._settings['debug'] = bool(value)

# 使用示例
config = Config()
print(config.debug_mode)  # 输出: False
config.debug_mode = True
print(config.debug_mode)  # 输出: True
```

## 常见问题与最佳实践

### 1. 命名约定

- 使用下划线前缀（如`_name`）来表示"私有"属性，这是Python中的约定。
- 属性名应具有描述性，清晰地表示其用途。

### 2. 避免副作用

```python
# 不好的示例
class Counter:
    def __init__(self):
        self._count = 0

    @property
    def count(self):
        self._count += 1  # 属性访问不应该改变对象状态
        return self._count

# 好的示例
class Counter:
    def __init__(self):
        self._count = 0

    @property
    def count(self):
        return self._count

    def increment(self):
        self._count += 1
        return self._count
```

### 3. 性能考虑

对于计算成本高的属性，考虑使用缓存机制：

```python
class CachedProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value

class DataProcessor:
    def __init__(self, data):
        self.data = data

    @CachedProperty
    def processed_data(self):
        # 假设这是一个耗时的处理过程
        print("执行数据处理...")
        return [x * 2 for x in self.data]

# 使用示例
processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.processed_data)  # 第一次访问会执行处理
print(processor.processed_data)  # 第二次访问直接返回缓存结果
```

### 4. 与@dataclass结合使用

```python
from dataclasses import dataclass

@dataclass
class Person:
    first_name: str
    last_name: str

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

# 使用示例
person = Person("王", "五")
print(person.full_name)  # 输出: 王 五
```

## 总结

@property装饰器是Python中一个强大而优雅的特性，它允许我们：

1. 将方法转换为属性，使代码更加Pythonic
2. 在不改变外部接口的情况下，为属性添加验证逻辑
3. 创建计算属性，动态计算值
4. 实现只读、可读写和可删除属性
5. 提高代码的可读性和可维护性

@property装饰器是Python面向对象编程的重要组成部分，掌握它的使用将使你的代码更加优雅和专业。在需要控制属性访问、添加验证逻辑或实现计算属性时，@property是一个很好的选择。
