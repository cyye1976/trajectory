package com.jinghui.api.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.jinghui.api.entity.Data;
import com.jinghui.api.entity.Model;
import com.jinghui.api.service.IDataService;
import com.jinghui.api.vo.Result;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.Part;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月07日 11:34
 */
@RestController
@Api(tags = "数据文件管理模块api")
@RequestMapping("/data")
@CrossOrigin//可以加在类上，也可以加到方法上
public class DataController {
    @Autowired
    IDataService dataService;

    @ApiOperation(value="数据记录-分页列表查询", notes="数据记录-分页列表查询")
    @GetMapping(value = "/list")
    public Result<?> queryPageList(@RequestParam(name = "pageNo", defaultValue = "1") Integer pageNo,
                                   @RequestParam(name = "pageSize", defaultValue = "10") Integer pageSize){
        Page<Data> page = new Page<>(pageNo, pageSize);
        IPage<Data> pageList = dataService.getPageList(page);
        return Result.ok(pageList);
    }

    @ApiOperation(value="数据记录-添加数据", notes="数据记录-添加数据")
    @PostMapping(value = "/add")
    public Result<?> addData(MultipartHttpServletRequest request) throws ServletException, IOException {
        MultiValueMap<String, MultipartFile> fileMap = request.getMultiFileMap();
        List<MultipartFile> multipartFiles = fileMap.get("files");
        String description = request.getParameter("description");
        Data data = new Data();
        data.setDescription(description);

        int result = dataService.addData(data, multipartFiles);
        if (result >= 1){
            return Result.ok();
        }else{
            return Result.fail("dataService-addData执行失败");
        }
    }

    @ApiOperation(value="数据记录-删除数据", notes="数据记录-删除数据")
    @PostMapping(value = "/delete/{id}")
    public Result<?> deleteData(@PathVariable int id){
        int result = dataService.deleteData(id);
        if (result >= 0){
            return Result.ok();
        }else {
            return Result.fail("dataService-deleteData执行失败");
        }
    }

    @ApiOperation(value="数据记录-修改数据", notes="数据记录-修改数据")
    @PostMapping(value = "/update")
    public Result<?> updateData(@RequestBody Data data){
        int result = dataService.updateData(data);
        if (result >= 0){
            return Result.ok();
        }else {
            return Result.fail("dataService-updateData执行失败");
        }
    }

    @ApiOperation(value="数据记录-展示图片", notes="数据记录-展示图片")
    @GetMapping(value = "/getPicture/{id}")
    public byte[] getPicture(@PathVariable int id) throws IOException {
        byte[] result = dataService.getPicture(id);
//        if (result.length > 0){
//            return Result.ok(result);
//        }else {
//            return Result.fail("dataService-getPicture执行失败");
//        }
        return result;
    }

    @GetMapping(value = "/getAllCount")
    public Result<?> getAllCount(){
        return Result.ok(dataService.getAllCount());
    }

}
