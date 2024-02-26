package com.bingzhuo.ship;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan(basePackages = "com.bingzhuo.ship.mapper")
public class ShipServerApplication {

	public static void main(String[] args) {
		SpringApplication.run(ShipServerApplication.class, args);
	}

}
