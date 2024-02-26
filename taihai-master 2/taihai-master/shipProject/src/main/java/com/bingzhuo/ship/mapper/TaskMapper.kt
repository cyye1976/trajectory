package com.bingzhuo.ship.mapper

import org.apache.ibatis.annotations.Param
import com.bingzhuo.ship.entity.po.Task

interface TaskMapper {
    fun getTaskById(@Param("taskId") taskId:Int):Task
    fun saveTask(@Param("task")task:Task)
    fun updateTask(@Param("task")task: Task)
}