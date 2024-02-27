package com.jinghui.api.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.jinghui.api.entity.Data;
import com.jinghui.api.entity.Model;
import com.jinghui.api.mapper.DataMapper;
import com.jinghui.api.mapper.ModelMapper;
import com.jinghui.api.service.IDataService;
import com.jinghui.api.service.IModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月07日 11:30
 */
@Service
public class DataServiceImpl extends ServiceImpl<DataMapper, Data> implements IDataService {
    @Autowired
    DataMapper dataMapper;

    @Value("${file.data-upload-dir}")
    private String uploadDir;

    @Override
    public Page<Data> getPageList(Page<Data> page) {
        return dataMapper.selectPage(page, null);
    }

    @Override
    public int deleteData(int id) {
        return dataMapper.deleteById(id);
    }

    @Override
    public int updateData(Data data) {
        return dataMapper.updateById(data);
    }

    @Override
    @Transactional
    public int addData(Data data, List<MultipartFile> files) {
        // 上传文件
        int count = 0;
        int consGroupId = -1;
        String uploadPath = uploadDir;
        try {
            for(MultipartFile file : files) {
                Data newData = new Data();
                newData.setDescription(data.getDescription());
                // 写入创建时间
                Date date = new Date();
                newData.setCreateTime(date);

                //生成uuid
                String uuid = UUID.randomUUID().toString().replaceAll("-", "");
                //得到上传时的文件名
                String filename = file.getOriginalFilename();
                //上传目录地址
                //1.1保存到项目指定目录
                //1.2 上传到相对路径 request.getSession().getServletContext().getRealPath("/")+"upload/";
                //1.2 此路径为tomcat下，可以输出看一看
                if (count == 0){
                    dataMapper.insert(newData);
                    consGroupId = newData.getId();
                    newData.setGroupId(consGroupId);
                    //如果目录不存在，自动创建文件夹
                    uploadPath = uploadDir + "\\"+ consGroupId + "\\";
                    File dir = new File(uploadPath);
                    if (!dir.exists()) {
                        dir.mkdir();
                    }
                    //保存文件对象 加上uuid是为了防止文件重名
                    File serverFile = new File( uploadPath + uuid + filename);
                    file.transferTo(serverFile);
                    // 写入模型文件路径
                    newData.setFileUrl(uploadPath + uuid + filename);
                    dataMapper.updateById(newData);
                }else {
                    //保存文件对象 加上uuid是为了防止文件重名
                    File serverFile = new File( uploadPath + uuid + filename);
                    file.transferTo(serverFile);
                    // 写入模型文件路径
                    newData.setFileUrl(uploadPath + uuid + filename);
                    newData.setGroupId(consGroupId);
                    dataMapper.insert(newData);
                }

                count++;
            }
            return 1;
        } catch (Exception e) {
            //打印错误堆栈信息
            e.printStackTrace();
            return 0;
        }
    }

    @Override
    public Data getDataById(int id) {
        return dataMapper.selectById(id);
    }

    @Override
    public byte[] getPicture(int id) throws IOException {
        Data data = dataMapper.selectById(id);
        File file = new File(data.getFileUrl());
        FileInputStream picInput = new FileInputStream(file);
        byte[] buffer = new byte[picInput.available()];
        picInput.read(buffer, 0, picInput.available());
        picInput.close();
        return buffer;

    }

    @Override
    public int addSingleData(Data data, MultipartFile file) throws IOException {
        Data newData = new Data();
        newData.setDescription(data.getDescription());
        // 写入创建时间
        Date date = new Date();
        newData.setCreateTime(date);

        //生成uuid
        String uuid = UUID.randomUUID().toString().replaceAll("-", "");
        //得到上传时的文件名
        String filename = file.getOriginalFilename();

        //如果目录不存在，自动创建文件夹
        dataMapper.insert(newData);
        int consGroupId = newData.getId();
        String uploadPath = uploadDir + "\\"+ consGroupId + "\\";
        File dir = new File(uploadPath);
        if (!dir.exists()) {
            dir.mkdir();
        }
        //保存文件对象 加上uuid是为了防止文件重名
        File serverFile = new File( uploadPath + uuid + filename);
        file.transferTo(serverFile);
        // 写入模型文件路径
        newData.setFileUrl(uploadPath + uuid + filename);
        newData.setGroupId(consGroupId);
        dataMapper.updateById(newData);

        return consGroupId;
    }

    @Override
    public int getAllCount() {
        return dataMapper.selectCount(null);
    }

    @Override
    public int getCountInCurrentDay() {
        QueryWrapper<Data> queryWrapper = new QueryWrapper<>();
        Date currentTime = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
        String rqq = formatter.format(currentTime);
        queryWrapper.apply("DATE(create_time) >= STR_TO_DATE('"+rqq+"00:00:00','%Y-%m-%d %H:%i:%s')");
        return dataMapper.selectCount(queryWrapper);
    }


}
