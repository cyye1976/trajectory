package com.jinghui.api.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.jinghui.api.entity.Model;
import com.jinghui.api.mapper.ModelMapper;
import com.jinghui.api.service.IModelService;
import com.jinghui.api.vo.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.UUID;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月06日 17:03
 */
@Service
@Component
public class ModelServiceImpl extends ServiceImpl<ModelMapper, Model> implements IModelService {

    @Autowired
    private ModelMapper modelMapper;

    @Value("${file.model-upload-dir}")
    private String uploadDir;

    @Override
    public Page<Model> getModelPageList(Page<Model> page) {
        return modelMapper.selectPage(page, null);
    }

    @Override
    public int addModel(Model model, MultipartFile file) {
        // 上传文件
        try {
            //生成uuid
            String uuid = UUID.randomUUID().toString().replaceAll("-", "");
            //得到上传时的文件名
            String filename=file.getOriginalFilename();
            //上传目录地址
            //1.1保存到项目指定目录
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
            // 写入模型文件路径
            model.setFileUrl(uploadDir+"\\"+uuid+filename);
            // 写入创建时间
            Date date = new Date();
            model.setCreateTime(date);

            return modelMapper.insert(model);
        } catch (Exception e) {
            //打印错误堆栈信息
            e.printStackTrace();
            return 0;
        }
    }

    @Override
    public int deleteModel(int id) {
        // 删除模型文件
        Model model = modelMapper.selectById(id);
        String fileUrl = model.getFileUrl();
        File file = new File(fileUrl);
        if (!file.delete()){
            return -1;
        }

        return modelMapper.deleteById(id);
    }

    @Override
    public int updateModel(Model model) {
        return modelMapper.updateById(model);
    }

    @Override
    public int getAllCount() {
        return modelMapper.selectCount(null);
    }

    @Override
    public int getCountInCurrentDay() {
        QueryWrapper<Model> queryWrapper = new QueryWrapper<>();
        Date currentTime = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
        String rqq = formatter.format(currentTime);
        queryWrapper.apply("DATE(create_time) >= STR_TO_DATE('"+rqq+"00:00:00','%Y-%m-%d %H:%i:%s')");
        return modelMapper.selectCount(queryWrapper);
    }




}
