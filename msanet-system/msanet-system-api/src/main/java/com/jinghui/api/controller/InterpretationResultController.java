package com.jinghui.api.controller;

import com.alibaba.fastjson.JSONObject;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.jinghui.api.entity.InterpretationResult;
import com.jinghui.api.service.IInterpretationResultService;
import com.jinghui.api.vo.Result;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.sql.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月09日 10:37
 */
@RestController
@Api(tags = "数据解译模块api")
@RequestMapping("/interpretation")
@CrossOrigin//可以加在类上，也可以加到方法上
public class InterpretationResultController {
    @Autowired
    IInterpretationResultService interpretationResultService;

    @ApiOperation(value="数据解译-分页列表查询", notes="数据解译-分页列表查询")
    @GetMapping(value = "/list")
    public Result<?> queryPageList(@RequestParam(name = "pageNo", defaultValue = "1") Integer pageNo,
                                   @RequestParam(name = "pageSize", defaultValue = "10") Integer pageSize){
        Page<InterpretationResult> page = new Page<>(pageNo, pageSize);
        IPage<InterpretationResult> pageList = interpretationResultService.getPageList(page);
        return Result.ok(pageList);
    }

    @ApiOperation(value="数据解译-结果删除", notes="数据解译-结果删除")
    @PostMapping(value = "/delete/{id}")
    public Result<?> deleteInterpretationResult(@PathVariable int id){
        int result = interpretationResultService.deleteInterpretationResult(id);
        if (result >= 0){
            return Result.ok();
        }else {
            return Result.fail("interpretationResultService-deleteInterpretationResult执行失败");
        }
    }

    @ApiOperation(value="数据解译-分析数据", notes="数据解译-分析数据")
    @PostMapping(value = "/analyseData/{dataId}/{modelId}")
    public Result<?> analyseData(@PathVariable int dataId, @PathVariable int modelId){
        int result = interpretationResultService.analyseData(dataId, modelId);
        if (result >= 0){
            return Result.ok(result);
        }else {
            return Result.fail("interpretationResultService-analyseData执行失败");
        }
    }

    @ApiOperation(value="数据解译-获取绘制信息", notes="数据解译-获取绘制信息")
    @PostMapping(value = "/getVisMessage/{id}")
    public Result<?> getVisMessage(@PathVariable int id){
        try {
            return Result.ok(interpretationResultService.getVisMessage(id));
        } catch (Exception e){
            return Result.fail("interpretationResultService-getVisMessage执行失败");
        }
    }

    @ApiOperation(value="数据解译-分析数据并可视化", notes="数据解译-分析数据并可视化")
    @PostMapping(value = "/analyseAndVisualizeData")
    public Result<?> analyseAndVisualizeData(HttpServletRequest request, MultipartFile file) throws IOException {
        int modelId = Integer.parseInt(request.getParameter("modelId"));
        String[] visOptionsStr = request.getParameter("visOptions").split(",");
        List<Integer> visOptions = Arrays.asList(visOptionsStr).stream().mapToInt(Integer::parseInt).boxed().collect(Collectors.toList());
        JSONObject result = interpretationResultService.analyseAndVisualizeData(modelId, file, visOptions);
        return Result.ok(result);
    }

    @ApiOperation(value="数据解译-通过路径读取图片", notes="数据解译-通过路径读取图片")
    @GetMapping(value = "/getFileURLPictures/{fileName}")
    public byte[] getFileURLPictures(@PathVariable String fileName) throws IOException {
        return interpretationResultService.getFileURLPictures(fileName);
    }
}
