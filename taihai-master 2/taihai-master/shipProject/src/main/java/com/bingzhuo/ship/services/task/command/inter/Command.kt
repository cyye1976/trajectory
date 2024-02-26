package com.bingzhuo.ship.services.task.command.inter

import java.util.concurrent.TimeUnit

interface Command<T> {
    fun execute():T
    fun getPrefix():String
    fun getAdditionalParams():Array<String>
    fun getCommand():String
    fun getTimeOut():Long
    fun getTimeUnit():TimeUnit
    fun dealResult(cmdOut:String):T
    fun getPyFilePathPrifix():String
}