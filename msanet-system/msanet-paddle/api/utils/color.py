def RGB_to_Hex(RGB):
    # RGB: [255, 255 ,255]           # 将RGB格式划分开来
    color = '#'
    for num in RGB:
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()

    return color