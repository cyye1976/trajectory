package com.jinghui.api;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.jinghui.api.mapper")//扫描mapper文件夹
public class MsanetSystemApiApplication {

	public static void main(String[] args) {
		SpringApplication.run(MsanetSystemApiApplication.class, args);
	}

}
