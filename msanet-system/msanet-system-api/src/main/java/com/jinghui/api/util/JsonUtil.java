package com.jinghui.api.util;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;

import java.io.*;

public class JsonUtil {
    public static JSONObject readJsonFile(String url){
        String jsonStr = "";
        try {
            File jsonFile = new File(url);
            FileReader fileReader = new FileReader(jsonFile);
            Reader reader = new InputStreamReader(new FileInputStream(jsonFile),"utf-8");
            int ch = 0;
            StringBuffer sb = new StringBuffer();
            while ((ch = reader.read()) != -1) {
                sb.append((char) ch);
            }
            fileReader.close();
            reader.close();
            jsonStr = sb.toString();
            JSONObject jobj = JSON.parseObject(jsonStr);
            return jobj;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
