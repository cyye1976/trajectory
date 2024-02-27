package com.jinghui.api.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.jinghui.api.entity.Model;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Component
public interface IModelService extends IService<Model> {

    /**
     * 获取所有模型表中的记录（分页返回）
     * @return
     */
    Page<Model> getModelPageList(Page<Model> page);

    /**
     * 添加模型上传记录
     * @param model
     */
    int addModel(Model model, MultipartFile file);

    /**
     * 删除模型上传记录
     * @param id
     * @return
     */
    int deleteModel(int id);

    /**
     * 修改模型上传记录
     * @param model
     * @return
     */
    int updateModel(Model model);

    /**
     * 获取所有模型总数
     * @return
     */
    int getAllCount();

    /**
     * 获取当日新增模型总数
     * @return
     */
    int getCountInCurrentDay();


}
