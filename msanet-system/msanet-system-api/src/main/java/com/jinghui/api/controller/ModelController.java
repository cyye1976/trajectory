package com.jinghui.api.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.jinghui.api.entity.Model;
import com.jinghui.api.service.IModelService;
import com.jinghui.api.vo.Result;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;

import java.io.File;
import java.util.*;
import javax.servlet.http.HttpServletRequest;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月06日 17:20
 */
@Api(tags = "模型管理api")
@RestController
@RequestMapping("/model")
@CrossOrigin//可以加在类上，也可以加到方法上
public class ModelController {
    @Autowired
    private IModelService modelService;
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @ApiOperation(value="模型记录-分页列表查询", notes="模型记录-分页列表查询")
    @GetMapping(value = "/list")
    public Result<?> queryPageList(@RequestParam(name = "pageNo", defaultValue = "1") Integer pageNo,
                                    @RequestParam(name = "pageSize", defaultValue = "10") Integer pageSize){
        Page<Model> page = new Page<>(pageNo, pageSize);
        IPage<Model> pageList = modelService.getModelPageList(page);
        return Result.ok(pageList);
    }

    @ApiOperation(value = "模型记录-添加模型", notes = "模型记录-添加模型")
    @PostMapping(value = "/add")
    public Result<?> addModel(HttpServletRequest request, MultipartFile file){
        Model model = new Model();
        model.setName(request.getParameter("name"));
        model.setDescription(request.getParameter("description"));

        int result = modelService.addModel(model, file);
        if(result >= 1){
            // 模型记录插入成功
            return Result.ok();
        }else {
            // 模型记录插入失败
            return Result.fail("modelService-addModel执行失败");
        }
    }

    @ApiOperation(value = "模型记录-删除模型", notes = "模型记录-删除模型")
    @GetMapping(value = "/delete/{id}")
    public Result<?> deleteModel(@PathVariable int id){
        int result = modelService.deleteModel(id);
        if (result >= 0){
            return Result.ok();
        }else {
            return Result.fail("modelService-deleteModel执行失败");
        }
    }

    @ApiOperation(value = "模型记录-修改模型", notes = "模型记录-修改模型")
    @PostMapping(value = "/update")
    public Result<?> updateModel(@RequestBody Model model){
        int result = modelService.updateModel(model);
        if (result >= 0){
            return Result.ok();
        }else {
            return Result.fail("modelService-updateModel执行失败");
        }
    }

    @PostMapping(value = "/uploadModelFile")
    public Result<?> uploadModelFile(HttpServletRequest request,MultipartFile file){
        try {
            //生成uuid
            String uuid = UUID.randomUUID().toString().replaceAll("-", "");
            //得到上传时的文件名
            String filename=file.getOriginalFilename();
            //上传目录地址
            //1.1保存到项目指定目录
            String uploadDir="D:/Developer/Workplace/important/main/msanet-system-api/src/main/resources/static";
            //1.2 上传到相对路径 request.getSession().getServletContext().getRealPath("/")+"upload/";
            //1.2 此路径为tomcat下，可以输出看一看

            //如果目录不存在，自动创建文件夹
            File dir=new File(uploadDir);
            if(!dir.exists()){
                dir.mkdir();
            }
            //保存文件对象 加上uuid是为了防止文件重名
            File serverFile=new File(uploadDir+"\\"+uuid+filename);
            file.transferTo(serverFile);
        } catch (Exception e) {
            //打印错误堆栈信息
            e.printStackTrace();
            return Result.fail("上传失败");
        }

        return Result.ok();
    }

    @GetMapping(value = "/getAllCount")
    public Result<?> getAllCount(){
        return Result.ok(modelService.getAllCount());
    }
}
