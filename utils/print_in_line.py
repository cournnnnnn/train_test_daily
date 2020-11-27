import time
scale = 50
print("这里是进度读取区域".center(62,"#"))
start = time.perf_counter()
for i in range(scale+1):
    a = "*" * i
    b = "." * (scale-i)
    c = (i/scale) * 100
    dur = time.perf_counter() - start
    print(("\r"+"#"*2+"{:^3.0f}%[{}->{}]{:.2f}s"+"#"*2).format(c,a,b,dur),end="")
    time.sleep(0.1 )
print("\n"+"这里是进度读取区域".center(62,"#"))