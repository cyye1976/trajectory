package com.jinghui.api.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.jinghui.api.entity.Data;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public interface IDataService extends IService<Data> {

    /**
     * 获取数据文件上传记录（分页查询）
     * @param page
     * @return
     */
    Page<Data> getPageList(Page<Data> page);

    /**
     * 删除数据文件上传记录
     * @param id
     * @return
     */
    int deleteData(int id);

    /**
     * 修改数据文件上传记录
     * @param data
     * @return
     */
    int updateData(Data data);

    /**
     * 添加数据文件上传记录
     * @param data
     * @return
     */
    int addData(Data data, List<MultipartFile> files);

    /**
     * 通过ID查询数据文件记录
     * @param id
     * @return
     */
    Data getDataById(int id);

    /**
     * 获取图片
     */
    byte[] getPicture(int id) throws IOException;

    /**
     * 单个上传数据
     * @return 返回数据ID号
     */
    int addSingleData(Data data, MultipartFile file) throws IOException;

    /**
     * 获取数据总量
     * @return
     */
    int getAllCount();

    /**
     * 获取当日新增数据总数
     * @return
     */
    int getCountInCurrentDay();
}
