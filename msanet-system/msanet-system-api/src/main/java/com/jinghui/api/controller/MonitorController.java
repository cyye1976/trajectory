package com.jinghui.api.controller;

import com.alibaba.fastjson.JSONObject;
import com.jinghui.api.dto.ModelUtilization5CountDto;
import com.jinghui.api.service.IDataService;
import com.jinghui.api.service.IInterpretationResultService;
import com.jinghui.api.service.IModelService;
import com.jinghui.api.vo.Result;
import io.swagger.annotations.Api;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Api(tags = "平台监控api")
@RestController
@RequestMapping("/monitor")
@CrossOrigin//可以加在类上，也可以加到方法上
public class MonitorController {

    @Autowired
    IDataService dataService;

    @Autowired
    IModelService modelService;

    @Autowired
    IInterpretationResultService interpretationResultService;

    @GetMapping("/getAllMonitorData")
    public Result<?> getAllMonitorData(){
        int modelCount = modelService.getAllCount();
        int dataCount = dataService.getAllCount();
        int interpretationCount = interpretationResultService.getAllCount();
        int modelCountCurrentDay = modelService.getCountInCurrentDay();
        int dataCountCurrentDay = dataService.getCountInCurrentDay();
        int interpretationCountCurrentDay = interpretationResultService.getCountInCurrentDay();
        List<ModelUtilization5CountDto> modelUtilization5CountDtos = interpretationResultService.getModelUtilizationRatio();
        JSONObject ship7Count = interpretationResultService.getShip7Count();

        JSONObject postData = new JSONObject();
        postData.put("modelCount", modelCount);
        postData.put("dataCount", dataCount);
        postData.put("interpretationCount", interpretationCount);
        postData.put("modelCountCurrentDay", modelCountCurrentDay);
        postData.put("dataCountCurrentDay", dataCountCurrentDay);
        postData.put("interpretationCountCurrentDay", interpretationCountCurrentDay);
        postData.put("modelUtilization5Count", modelUtilization5CountDtos);
        postData.put("ship7count", ship7Count);

        return Result.ok(postData);
    }

}
