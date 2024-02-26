package com.bingzhuo.ship.services.task

import org.springframework.stereotype.Component
import java.lang.RuntimeException
import java.util.*
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.collections.ArrayList

@Component
class TaskStatusManagerImpl:TaskSatatusManager {
    //需要强一致性
    private val taskContinar:MutableList<Int> = CopyOnWriteArrayList()

    override fun regesitTasking(taskId: Int): Boolean {
        if(!isTasking(taskId)){
            synchronized(this){
                if(!isTasking(taskId))
                    return taskContinar.add(taskId)
                throw RuntimeException("该任务已被注册")
            }
        }else{
            throw RuntimeException("该任务已被注册")
        }
    }

    override fun deleteTasking(taskId: Int): Boolean {
        return taskContinar.remove(taskId)
    }

    override fun isTasking(taskId: Int): Boolean {
        return taskContinar.contains(taskId)
    }
}