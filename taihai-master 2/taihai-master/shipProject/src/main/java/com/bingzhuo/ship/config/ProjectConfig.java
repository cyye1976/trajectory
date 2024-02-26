package com.bingzhuo.ship.config;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

@Component
public class ProjectConfig {
	
	private Environment env;
	@Autowired
	public ProjectConfig(Environment env) {
		this.env = env;
	}
	public String getValue(String key) {
		return (String) this.env.getProperty(key);
	}
}
