#!/usr/bin/ruby
class Sample
    def hello(string)
       puts "Hello Ruby! #{string}"
    end
end

object = Sample.new
object.hello("I am object")

class Customer
    $global_variable = "I am global variable" # 全局变量
    CONSTANT = 1.2345 # 常量
    @@no_of_customers=0 # 类变量
    def initialize(id, name, addr) # 构造函数, 在new时自动调用
       @cust_id=id # 实例变量
       @cust_name=name # 实例变量
       @cust_addr=addr # 实例变量
    end
    def display_details()
        puts "Customer id #@cust_id"
        puts "Customer name #@cust_name"
        puts "Customer address #@cust_addr"
    end
    def total_no_of_customers()
        @@no_of_customers += 1
        puts "Total number of customers: #@@no_of_customers"
    end
end

cust1=Customer.new("1", "John", "Wisdom Apartments, Ludhiya")
cust2=Customer.new("2", "Poul", "New Empire road, Khandala")
puts cust1
puts cust2
cust1.display_details()
cust1.total_no_of_customers()
cust2.display_details()
cust2.total_no_of_customers()

puts cust1.inspect # 打印对象的信息
puts cust1.to_s # 打印对象的字符串