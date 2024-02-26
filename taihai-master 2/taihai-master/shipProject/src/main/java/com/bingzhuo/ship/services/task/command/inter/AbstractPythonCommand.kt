package com.bingzhuo.ship.services.task.command.inter

import com.bingzhuo.ship.services.task.command.tool.CommandTool

abstract class AbstractPythonCommand<T>:Command<T> {
    final override fun execute(): T {
        val command = getPrefix() + " " + getCommand() + " " + getAdditionalParams().joinToString ( " "){e->e}
        println(command)
        val result = CommandTool.execCommand(command,getTimeOut(),getTimeUnit())
        return dealResult(result)
    }
}