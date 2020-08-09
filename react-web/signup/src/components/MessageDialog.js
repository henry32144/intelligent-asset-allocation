import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogContent from '@material-ui/core/DialogContent';
import Typography from '@material-ui/core/Typography';


const useStyles = makeStyles((theme) => ({
  message: {
    textAlign: "center",
    margin: theme.spacing(0, 0, 2),
  }
}));

export default function MessageDialog(props) {

  const classes = useStyles();

  return (
    <div>
      <Dialog open={props.isOpen} onClose={props.handleClose} aria-labelledby="form-dialog-title">
        {
          props.title != null && 
          <DialogTitle>{props.title}</DialogTitle>
        }
        <DialogContent>
          <Typography className={classes.message}>
            {props.message}
          </Typography>
        </DialogContent>
      </Dialog>
    </div>
  );
}