package com.bingzhuo.ship.services.task

interface TaskSatatusManager {
    fun regesitTasking(taskId:Int):Boolean
    fun deleteTasking(taskId:Int):Boolean
    fun isTasking(taskId:Int):Boolean
}