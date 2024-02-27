package com.jinghui.api.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import io.swagger.annotations.ApiParam;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Description: TODO
 * @author: scott
 * @date: 2022年01月06日 16:23
 */
@Data
@ApiModel
public class Model {
    @TableId(type = IdType.AUTO)
    @ApiModelProperty("ID")
    private int Id;
    @ApiModelProperty("名称")
    private String name;
    @ApiModelProperty("模型文件路径")
    private String fileUrl;
    @ApiModelProperty("描述")
    private String description;
    @ApiModelProperty("创建时间")
    @JsonFormat(timezone = "GMT+8",pattern = "yyyy-MM-dd HH:mm:ss")
    @DateTimeFormat(pattern="yyyy-MM-dd HH:mm:ss")
    private Date createTime;
}
