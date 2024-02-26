package com.bingzhuo.ship.controller;

import com.bingzhuo.ship.entity.dto.TaskDto;
import com.bingzhuo.ship.entity.vo.ExtractResult;
import com.bingzhuo.ship.entity.vo.SplitResult;
import com.bingzhuo.ship.entity.vo.restful.Response;
import com.bingzhuo.ship.services.task.TaskService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import com.bingzhuo.ship.entity.po.Task;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Api(description = "船舶轨迹聚类相关API")
@RestController("/task")
public class TaskController {
    @Autowired
    private TaskService taskService;

    @PostMapping("/creat")
    @ApiOperation("上传数据集并新建任务")
    public Response<Task> creatTask(MultipartFile file){
        return Response.success("创建成功",taskService.creatTask(new TaskDto(file)));
    }

    @PostMapping("/feature_extract")
    @ApiOperation("特征提取")
    public Response<List<ExtractResult>> extractFeature(int taskId){
        List<String> resultList = taskService.featureExtract(taskId);
        return Response.success("创建成功",resultList.stream().map(e->new ExtractResult(e)).collect(Collectors.toList()));
    }

    @PostMapping("/split")
    @ApiOperation("数据集切分")
    public Response<SplitResult> split(int taskId){
        SplitResult sr = new SplitResult(taskService.traceSplit(taskId));
        return Response.success("切分成功", sr);
    }

    @PostMapping("/cluster")
    @ApiOperation("数据集切分")
    public Response cluster(int taskId){
        Map<String,String> result = taskService.gmmCluster(taskId);
        return Response.success("切分成功", result);
    }
}
