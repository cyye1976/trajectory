package com.jinghui.api.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;


/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月07日 9:56
 */
@Data
@ApiModel
public class InterpretationResult {
    @TableId(type = IdType.AUTO)
    @ApiModelProperty("ID")
    private int Id;
    @ApiModelProperty("数据ID号")
    private int dataId;
    @ApiModelProperty("模型ID号")
    private int modelId;
    @ApiModelProperty("数据解译结果JSON文件保存路径")
    private String jsonUrl;
    @ApiModelProperty("创建时间")
    @JsonFormat(timezone = "GMT+8",pattern = "yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat(pattern="yyyy-MM-dd HH:mm:ss")
    private Date createTime;
}
