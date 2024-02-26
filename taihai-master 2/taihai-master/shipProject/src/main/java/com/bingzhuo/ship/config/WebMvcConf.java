package com.bingzhuo.ship.config;

import javax.servlet.MultipartConfigElement;

import org.springframework.boot.web.servlet.MultipartConfigFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.unit.DataSize;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebMvcConf implements WebMvcConfigurer {

    //配置跨域
    @Override
    public void addCorsMappings(CorsRegistry registry){
        registry.addMapping("/**")
                .allowedHeaders("*")
                .allowedOrigins("*")
                .allowedMethods("*")
                .allowCredentials(true);
    }

    //拦截器
    @Override
    public void addInterceptors(InterceptorRegistry registry){


    }
    @Bean
    public MultipartConfigElement multipartConfigElement() {
       MultipartConfigFactory factory = new MultipartConfigFactory();
       //  单个数据大小
       factory.setMaxFileSize(DataSize.parse("60MB"));
       /// 总上传数据大小
       factory.setMaxRequestSize(DataSize.parse("80MB"));
       return factory.createMultipartConfig();
    }
}
